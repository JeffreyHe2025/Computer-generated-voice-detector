import tensorflow as tf
import keras
from keras import layers
import coremltools as ct
from tensorflow.keras.models import load_model


# Custom layer used by the LCNN architecture. Must be registered so Keras
# can deserialize the .keras file (which only stores the class name "MFM").
@keras.saving.register_keras_serializable()
class MFM(layers.Layer):
    """Max-Feature-Map: split channels in half, take elementwise max."""
    def call(self, inputs):
        n = inputs.shape[-1] // 2
        return tf.maximum(inputs[..., :n], inputs[..., n:])

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [input_shape[-1] // 2])


keras_model = load_model(
    "trained_voice_detector_lcnn.keras",
    custom_objects={"MFM": MFM},
)
keras_model.summary()

# LCNN expects (168, 94, 1) — mel-spectrogram (128) + MFCC (40) concatenated.
@tf.function(input_signature=[
    tf.TensorSpec(shape=(1, 168, 94, 1), dtype=tf.float32, name="features")
])
def serve(features):
    return keras_model(features)

concrete_fn = serve.get_concrete_function()

mlmodel = ct.convert(
    [concrete_fn],
    source="tensorflow",
    inputs=[ct.TensorType(name="features", shape=(1, 168, 94, 1))],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
)

mlmodel.save("VoiceDetector_lcnn.mlpackage")
print("Done. Saved VoiceDetector_lcnn.mlpackage")