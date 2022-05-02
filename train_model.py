# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from sklearn.utils import shuffle

import datetime

import tensorflow as tf
import numpy as np

import build_model
import load_and_vectorize_data

FLAGS = None


def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.
    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = load_and_vectorize_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val = load_and_vectorize_data.ngram_vectorize(
        train_texts, train_labels, val_texts)

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
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

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

    # Save model.
    model.save('weights/model_mlp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':
    train_texts, train_labels = load_and_vectorize_data.load_data_into_lists(load_and_vectorize_data.train_filepath)
    print(train_texts, train_labels)

    val_test_texts, val_test_labels = load_and_vectorize_data.load_data_into_lists(load_and_vectorize_data.test_filepath)
    print(val_test_texts, val_test_labels)

    # shuffle the texts and labels
    shuffle_random_seed = 42
    train_texts, train_labels = shuffle(train_texts, train_labels, random_state=shuffle_random_seed)
    # TODO make sure that the val and test set are always split in the same way
    val_test_texts, val_test_labels = shuffle(val_test_texts, val_test_labels, random_state=shuffle_random_seed)

    # split the val + test dataset into val dataset and test dataset
    val_texts, val_labels, test_texts, test_labels = load_and_vectorize_data.split_val_test_set(val_test_texts,
                                                                                                val_test_labels)

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    data = ((train_texts, train_labels), (val_texts, val_labels))

    history = train_ngram_model(data)
    print(history)




