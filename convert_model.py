import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Convert a Keras TF2 model (.keras or SavedModel) to .tflite')
parser.add_argument('-i', '--input', type=str, required=True,
                    help='path to the Keras model (.keras) or SavedModel directory')
parser.add_argument('-o', '--output', type=str,
                    default='converted_model.tflite',
                    help='path to output the .tflite model to')
parser.add_argument('--quantize', action='store_true',
                    help='enable dynamic-range quantization (no representative dataset needed)')

args = parser.parse_args()

# Load model (supports .keras file or SavedModel dir)
try:
    model = tf.keras.models.load_model(args.input)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
except Exception:
    # Fallback: try SavedModel directory
    converter = tf.lite.TFLiteConverter.from_saved_model(args.input)

# Optional dynamic-range quantization
if args.quantize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open(args.output, 'wb') as f:
    f.write(tflite_model)