import tensorflow as tf
import numpy as np
import collections
import argparse
import os

from build_model import mlp_model
import load_and_process_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--params_path', default='params.json',
                    help="Path to json file with parameters")


def load_model(filepath):
    # TODO make these parameters of the method (and/or be read from a config file)
    model = mlp_model(layers=2,
                      units=64,
                      dropout_rate=0.2,
                      input_shape=x_train.shape[1:],
                      num_classes=5)  # TODO

    loss = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.load_weights(filepath)
    return model


if __name__ == '__main__':

    args = parser.parse_args()
    assert os.path.isfile(args.params_path), "No json configuration file found at {}".format(args.params_path)
    params = utils.Params(args.params_path)

    train_filepath = os.path.join(params.data_dir, params.train_filename)
    val_test_filepath = os.path.join(params.data_dir, params.val_test_filename)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_and_process_data.load_train_val_test_datasets(train_filepath, val_test_filepath)

    x_train, x_test = load_and_process_data.ngram_vectorize(train_texts, train_labels, test_texts)
    _, x_val = load_and_process_data.ngram_vectorize(train_texts, train_labels, val_texts)

    filepath = 'weights/model_mlp_20220502_200753.h5'
    model = load_model(filepath)

    y_test_pred = model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    print(collections.Counter((y_test_pred == test_labels)))

    y_val_pred = model.predict(x_val)
    y_val_pred = np.argmax(y_val_pred, axis=1)
    print(collections.Counter((y_val_pred == val_labels)))

    y_train_pred = model.predict(x_train)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    print(collections.Counter((y_train_pred == train_labels)))

