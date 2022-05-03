import utils
import argparse
import os
import datetime

import tensorflow as tf
import numpy as np

import build_model
import load_and_process_data
import evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument('--params_path', default='params.json',
                    help="Path to json file with parameters")


def train_ngram_model(data, learning_rate=1e-3, epochs=1000, batch_size=128, layers=2,units=64, dropout_rate=0.2,
                      ngram_range=2, ngram_top_k=20000, ngram_token_mode="word", ngram_min_document_frequency=2):
    """Trains n-gram model on the given dataset.
    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        ngram_range: int, maximum size of an n-gram (e.g. for 2, we will have 1-grams and 2-grams).
        ngram_top_k: int, limit on the number of features of the ngram.
        ngram_token_mode: str, whether text should be split into word or character n-grams (either 'word' or 'char')
        ngram_min_document_frequency: int, minimum document/corpus frequency below which a token will be discarded
    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = load_and_process_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val = load_and_process_data.ngram_vectorize(train_texts, train_labels, val_texts,
                                           ngram_range=ngram_range,
                                           ngram_top_k=ngram_top_k,
                                           ngram_token_mode=ngram_token_mode,
                                           ngram_min_document_frequency=ngram_min_document_frequency)

    # Create model instance.
    model = build_model.mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])  # TODO add F1 score to metrics

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
    print('F1 micro score:', evaluate_model.evaluate_model_on_test_set(model))

    # Save model.
    model.save('weights/model_mlp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':

    args = parser.parse_args()
    assert os.path.isfile(args.params_path), "No json configuration file found at {}".format(args.params_path)
    params = utils.Params(args.params_path)

    train_filepath = os.path.join(params.data_dir, params.train_filename)
    val_test_filepath = os.path.join(params.data_dir, params.val_test_filename)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_and_process_data.load_train_val_test_datasets(train_filepath, val_test_filepath)

    data = ((train_texts, train_labels), (val_texts, val_labels))

    history = train_ngram_model(data, learning_rate=params.mlp_model_learning_rate, epochs=params.mlp_model_epochs,
                                batch_size=params.mlp_model_batch_size, layers=params.mlp_model_layers,
                                units=params.mlp_model_units, dropout_rate=params.mlp_model_dropout_rate,
                                ngram_range=params.ngram_range, ngram_top_k=params.ngram_top_k,
                                ngram_token_mode=params.ngram_token_mode,
                                ngram_min_document_frequency=params.ngram_min_document_frequency)
    print(history)




