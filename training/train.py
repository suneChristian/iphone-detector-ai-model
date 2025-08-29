import argparse
import os
import tensorflow as tf
from model import create_model
from utils import SimpleCOCODataGenerator as DataGenerator

def train_model(data_generator, trained_model_path, model_configuration, train_epochs):
    """Initiate, compile, train and save the model using TF2 (no sessions)."""

    # Initiate the model
    model = create_model(**model_configuration)

    # Compile and train the model
    model.compile(
        optimizer='adam',
        metrics=['binary_accuracy'],
        loss='binary_crossentropy'
    )

    model.summary()
    model.fit(data_generator, epochs=train_epochs)

    # Ensure output directory exists and save in Keras native format
    if not trained_model_path.endswith(('.keras', '.h5')):
        trained_model_path = trained_model_path + '.keras'
    os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
    model.save(trained_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a basic model.')
    parser.add_argument('-i', '--images', type=str, required=True,
                        help='path to the directory containing training \
                        images')
    parser.add_argument('-a', '--annotations', type=str, required=True,
                        help='path to the .json-file containing COCO instance \
                        annotations')
    parser.add_argument('--input-width', type=int, default=480,
                        help='The width of the model\'s input image')
    parser.add_argument('--input-height', type=int, default=270,
                        help='The height of the model\'s input image')
    parser.add_argument('-e', '--training-epochs', type=int, default=8,
                        help='number of training epochs')
    parser.add_argument('-t', '--tuning-epochs', type=int, default=4,
                        help='number of fine-tuning epochs')
    args = parser.parse_args()
    print('Using TensorFlow version: {}'.format(tf.__version__))

    data_generator = DataGenerator(args.images, args.annotations, batch_size=8,
                                   width=args.input_width, height=args.input_height)

    # verify data generator is created simply by printing the first batch
    print(data_generator.batch_size)

    trained_model_path = '/env/models/fp32_model/model.keras'
    quantized_model_path = '/env/models/qat_model/model'
    final_frozen_graph_path = '/env/models/trained_model.pb'

    train_epochs = args.training_epochs
    tune_epochs = args.tuning_epochs

    model_configuration = {'input_shape': (args.input_height, args.input_width, 3),
                        'n_blocks': 5, 'n_filters': 16}

    train_model(data_generator, trained_model_path, model_configuration, train_epochs)

