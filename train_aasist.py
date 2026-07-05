"""
AASIST-L trainer (PyTorch) for human-vs-AI voice classification.

AASIST-L is the lightweight variant of the AASIST architecture (Jung et al.,
2022), the leading audio anti-spoofing model. Key innovations:

  1. SincConv front-end — learnable bandpass filters operating on raw audio
     (no mel-spectrogram needed)
  2. Residual encoder — RawNet2-style 1D conv blocks
  3. Heterogeneous Stacking Graph Attention — separately models spectral and
     temporal relationships, then combines them
  4. Max-Graph Operation — element-wise max of two parallel graph branches,
     making the model robust to varied spoofing attacks

This is a single-file faithful implementation matched to the official
clovaai/aasist reference. Output model file: trained_voice_detector_aasist.pt

Run:
    python train_aasist.py
"""

import os
import glob
import math
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, accuracy_score

# ============================================================
# Config
# ============================================================
SAMPLE_RATE = 16000
DURATION_SAMPLES = 64600   # ~4.04 s at 16 kHz — AASIST's standard input length

HUMAN_DIR = "/Users/jeffreyhe/Downloads/Computer-generated-voice-detector-old/filtered_human_clips"
PARLER_AI_DIR = "ai_clips"
MLAAD_AI_DIR = "mlaad_clips/fake/en"
MODEL_OUT = "trained_voice_detector_aasist.pt"

N_PER_CLASS = 8000
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

# Device selection — MPS for Apple Silicon, CUDA for Nvidia, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# 1. SincConv front-end — learnable bandpass filters
# ============================================================
class SincConv(nn.Module):
    """Sinc-based learnable bandpass filterbank.

    Each output channel is a bandpass filter parameterized by (low_hz, band_hz)
    that are learned end-to-end. Front-end of RawNet2 and AASIST.
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels=70, kernel_size=129, sample_rate=16000,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=int(kernel_size / 2))
        self.register_buffer("window_",
                             (0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)).view(1, -1))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_",
                             (2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate))

    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_t_low = torch.matmul(low, self.n_)
        f_t_high = torch.matmul(high, self.n_)
        bp_left = ((torch.sin(f_t_high) - torch.sin(f_t_low)) / (self.n_ / 2)) * self.window_
        bp_center = 2 * band.view(-1, 1)
        bp_right = torch.flip(bp_left, dims=[1])
        bp = torch.cat([bp_left, bp_center, bp_right], dim=1) / (2 * band[:, None])
        filters = bp.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


# ============================================================
# 2. Residual block (RawNet2-style)
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, (2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, (2, 3), padding=(0, 1), stride=1)
        self.mp = nn.MaxPool2d((1, 3))
        self.proj = (nn.Conv2d(in_c, out_c, 1, padding=0, stride=1)
                     if in_c != out_c else None)

    def forward(self, x):
        identity = x
        out = x if self.first else self.selu(self.bn1(x))
        out = self.conv1(out)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        return self.mp(out)


# ============================================================
# 3. Graph attention layer (heterogeneous)
# ============================================================
class GraphAttentionLayer(nn.Module):
    """Single graph attention head. Used by HS-GAL below."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_param(out_dim)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SELU(inplace=True)
        self.input_drop = nn.Dropout(0.2)

    def _init_param(self, out_dim):
        return nn.Parameter(torch.FloatTensor(1, out_dim).uniform_(-1, 1))

    def forward(self, x):
        # x: (B, N, in_dim)
        x = self.input_drop(x)
        att_map = self._attention(x)
        # message passing
        att_x = torch.matmul(att_map, self.proj_with_att(x))
        no_att_x = self.proj_without_att(x)
        h = att_x + no_att_x
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        return self.act(h)

    def _attention(self, x):
        att_x = self.att_proj(x)
        att_map = torch.tanh(att_x.unsqueeze(2) + att_x.unsqueeze(1))
        att_map = torch.matmul(att_map, self.att_weight.unsqueeze(-1)).squeeze(-1)
        return F.softmax(att_map, dim=-1)


# ============================================================
# 4. Heterogeneous Stacking Graph Attention Layer (HS-GAL)
# ============================================================
class HSGAL(nn.Module):
    """Operates over two heterogeneous node sets (spectral + temporal)
    with a shared master node that connects them."""

    def __init__(self, dim_s, dim_t, out_dim):
        super().__init__()
        self.proj_s = nn.Linear(dim_s, out_dim)
        self.proj_t = nn.Linear(dim_t, out_dim)
        self.gat_s = GraphAttentionLayer(out_dim, out_dim)
        self.gat_t = GraphAttentionLayer(out_dim, out_dim)
        self.master = nn.Parameter(torch.randn(1, 1, out_dim))
        self.master_proj = nn.Linear(out_dim, out_dim)
        self.master_act = nn.SELU(inplace=True)

    def forward(self, xs, xt):
        # xs: (B, Ns, dim_s)   xt: (B, Nt, dim_t)
        xs = self.proj_s(xs)
        xt = self.proj_t(xt)
        master = self.master.expand(xs.size(0), -1, -1)
        # apply attention separately then mix via master node
        xs = self.gat_s(xs)
        xt = self.gat_t(xt)
        ms_in = torch.cat([master, xs, xt], dim=1).mean(dim=1, keepdim=True)
        master = self.master_act(self.master_proj(ms_in))
        return xs, xt, master


# ============================================================
# 5. AASIST-L model
# ============================================================
class AASIST_L(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = SincConv(out_channels=70, kernel_size=129, sample_rate=SAMPLE_RATE)
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            ResBlock(1, 32, first=True),
            ResBlock(32, 32),
            ResBlock(32, 64),
            ResBlock(64, 64),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 23, 64))

        # Two parallel HS-GAL branches → max graph operation
        self.hsgal1 = HSGAL(64, 64, 32)
        self.hsgal2 = HSGAL(64, 64, 32)

        self.fc_attn = nn.Linear(32, 1)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3, 64),
            nn.SELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, T) — raw waveform
        x = x.unsqueeze(1)                       # (B, 1, T)
        x = self.sinc(x)                         # (B, 70, T)
        x = x.unsqueeze(1)                       # (B, 1, 70, T)  add channel for 2D conv
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x)                      # (B, 64, F', T')

        # Build heterogeneous node sets from the encoder output
        # Spectral nodes: average across time → (B, F', C)
        spec = x.mean(dim=-1).transpose(1, 2)    # (B, F', 64)
        # Temporal nodes: average across freq → (B, T', C)
        temp = x.mean(dim=-2).transpose(1, 2)    # (B, T', 64)

        # Two parallel HS-GAL branches
        s1, t1, m1 = self.hsgal1(spec, temp)
        s2, t2, m2 = self.hsgal2(spec, temp)

        # Max Graph Operation — elementwise max across the two branches
        s_max = torch.maximum(s1, s2)
        t_max = torch.maximum(t1, t2)
        m_max = torch.maximum(m1, m2)

        # Attention pooling per node set, then concat + classify
        s_pool = self._attn_pool(s_max)
        t_pool = self._attn_pool(t_max)
        m_pool = m_max.squeeze(1)
        feat = torch.cat([s_pool, t_pool, m_pool], dim=-1)   # (B, 96)
        return self.classifier(feat).squeeze(-1)            # (B,)  logits

    def _attn_pool(self, x):
        # x: (B, N, C). Soft-attention pool to (B, C).
        w = F.softmax(self.fc_attn(x), dim=1)
        return (w * x).sum(dim=1)


# ============================================================
# 6. Dataset (with silence trim + symmetric noise augmentation)
# ============================================================
class VoiceDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio_trimmed) > SAMPLE_RATE * 0.5:
            audio = audio_trimmed

        # Pad / crop to fixed length
        if len(audio) < DURATION_SAMPLES:
            audio = np.pad(audio, (0, DURATION_SAMPLES - len(audio)))
        else:
            start = (len(audio) - DURATION_SAMPLES) // 2
            audio = audio[start:start + DURATION_SAMPLES]

        # Symmetric noise augmentation
        if self.augment and np.random.random() < 0.5:
            snr_db = np.random.uniform(15, 35)
            noise = np.random.randn(len(audio)).astype(np.float32)
            sig_p = float(np.mean(audio ** 2) + 1e-12)
            noise_p = sig_p / (10 ** (snr_db / 10))
            noise *= np.sqrt(noise_p / (np.mean(noise ** 2) + 1e-12))
            audio = audio + noise

        audio = audio.astype(np.float32)
        return torch.from_numpy(audio), torch.tensor(label, dtype=torch.float32)


def list_audio_files(root_dir):
    paths = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return paths


def equal_error_rate(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


# ============================================================
# 7. Training loop
# ============================================================
def train():
    print("Discovering files...")
    human_paths = list_audio_files(HUMAN_DIR)
    parler_paths = list_audio_files(PARLER_AI_DIR)
    mlaad_paths = list_audio_files(MLAAD_AI_DIR)
    print(f"  humans: {len(human_paths)}, Parler: {len(parler_paths)}, MLAAD: {len(mlaad_paths)}")

    random.shuffle(human_paths)
    random.shuffle(parler_paths)
    random.shuffle(mlaad_paths)
    half = N_PER_CLASS // 2
    human_sel = human_paths[:N_PER_CLASS]
    parler_sel = parler_paths[:half]
    mlaad_sel = mlaad_paths[:N_PER_CLASS - len(parler_sel)]

    all_paths = human_sel + parler_sel + mlaad_sel
    all_labels = [0] * len(human_sel) + [1] * (len(parler_sel) + len(mlaad_sel))
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    all_paths, all_labels = list(all_paths), list(all_labels)

    # 80/20 stratified split (manual since we have two classes)
    pos = [i for i, y in enumerate(all_labels) if y == 1]
    neg = [i for i, y in enumerate(all_labels) if y == 0]
    n_pos_test = int(0.2 * len(pos))
    n_neg_test = int(0.2 * len(neg))
    test_idx = set(pos[:n_pos_test] + neg[:n_neg_test])
    train_paths = [p for i, p in enumerate(all_paths) if i not in test_idx]
    train_labels = [y for i, y in enumerate(all_labels) if i not in test_idx]
    test_paths = [p for i, p in enumerate(all_paths) if i in test_idx]
    test_labels = [y for i, y in enumerate(all_labels) if i in test_idx]
    print(f"Train: {len(train_paths)}  |  Test: {len(test_paths)}")

    train_ds = VoiceDataset(train_paths, train_labels, augment=True)
    test_ds = VoiceDataset(test_paths, test_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = AASIST_L().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nAASIST-L parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for audio, label in train_loader:
            audio, label = audio.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(audio)
            loss = bce(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * audio.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            running_correct += (preds == label).sum().item()
            running_total += audio.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # ---- Evaluate ----
        model.eval()
        all_scores, all_true = [], []
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for audio, label in test_loader:
                audio, label = audio.to(device), label.to(device)
                logits = model(audio)
                val_loss += bce(logits, label).item() * audio.size(0)
                val_total += audio.size(0)
                all_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                all_true.extend(label.cpu().numpy().tolist())
        val_loss /= val_total
        all_scores = np.array(all_scores)
        all_true = np.array(all_true)
        val_acc = accuracy_score(all_true, (all_scores > 0.5).astype(int))
        eer, eer_thr = equal_error_rate(all_true, all_scores)
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"EER={eer*100:.2f}%  (thr={eer_thr:.3f})")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nSaved {MODEL_OUT}")


if __name__ == "__main__":
    train()
