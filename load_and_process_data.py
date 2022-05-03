# Parts of the code taken from Google tutorial on text classification:
# https://developers.google.com/machine-learning/guides/text-classification

import pickle
import os
import argparse

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import shuffle

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--params_path', default='params.json',
                    help="Path to json file with parameters")

# TODO make this not hard-coded but extracted from the data
LABEL_STRINGS_TO_INTEGERS = {'italian': 0, 'mexican': 1, 'southern_us': 2, 'indian': 3, 'chinese': 4}


def load_data_into_lists(filepath):
    texts = []
    labels = []
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        for data_sample in data:
            text = data_sample['ingredients']
            label_str = data_sample['cuisine']
            label_int = LABEL_STRINGS_TO_INTEGERS[label_str]
            texts.append(text)
            labels.append(label_int)
    return texts, labels


def ngram_vectorize(train_texts, train_labels, test_texts, ngram_range=2, ngram_top_k=20000, ngram_token_mode="word",
                    ngram_min_document_frequency=2):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        test_texts: list, test text strings.
        ngram_range: int, maximum size of an n-gram (e.g. for 2, we will have 1-grams and 2-grams).
        ngram_top_k: int, limit on the number of features.
        ngram_token_mode: str, whether text should be split into word or character n-grams (either 'word' or 'char')
        ngram_min_document_frequency: int, minimum document/corpus frequency below which a token will be discarded

    # Returns
        x_train, x_test: vectorized training and test texts
    """
    # range (inclusive) of n-gram sizes for tokenizing text
    ngram_range = tuple(range(1, 1 + ngram_range))

    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': ngram_range,  # by default we will use 1-grams and 2-grams
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': ngram_token_mode,  # by default we will split text into word tokens
            'min_df': ngram_min_document_frequency,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize test texts.
    x_test = vectorizer.transform(test_texts)

    # Select top 'k' of the vectorized features.
    # TODO check if the label variable should be one-hot encoded
    # TODO use some other scoring function?
    selector = SelectKBest(f_classif, k=min(ngram_top_k, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    return x_train, x_test


def get_num_classes(labels):
    """Gets the total number of classes.
    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)
    # Returns
        int, total number of classes.
    # Raises
        ValueError: if any label value in the range(0, num_classes - 1)
            is missing or if number of classes is <= 1.
    """
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}. Please make sure you have '
                         'at least one sample for every label value '
                         'in the range(0, {max_class})'.format(
                            missing_classes=missing_classes,
                            max_class=num_classes - 1))

    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes


def split_val_test_set(val_test_texts, val_test_labels):
    val_set_count = len(val_test_labels) // 2
    val_texts = val_test_texts[: val_set_count]
    val_labels = val_test_labels[: val_set_count]
    test_texts = val_test_texts[val_set_count:]
    test_labels = val_test_labels[val_set_count:]
    return val_texts, val_labels, test_texts, test_labels


def load_train_val_test_datasets(train_filepath, val_test_filepath, shuffle_random_seed=42):
    train_texts, train_labels = load_data_into_lists(train_filepath)
    val_test_texts, val_test_labels = load_data_into_lists(val_test_filepath)

    # shuffle the texts and labels
    train_texts, train_labels = shuffle(train_texts, train_labels, random_state=shuffle_random_seed)
    # TODO make sure that the val and test set are always split in the same way
    val_test_texts, val_test_labels = shuffle(val_test_texts, val_test_labels, random_state=shuffle_random_seed)

    # split the val + test dataset into val dataset and test dataset
    val_texts, val_labels, test_texts, test_labels = split_val_test_set(val_test_texts, val_test_labels)

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


if __name__ == '__main__':

    args = parser.parse_args()
    assert os.path.isfile(args.params_path), "No json parameters file found at {}".format(args.params_path)
    params = utils.Params(args.params_path)

    train_filepath = os.path.join(params.data_dir, params.train_filename)
    val_test_filepath = os.path.join(params.data_dir, params.val_test_filename)


    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_train_val_test_datasets(
                                                                                train_filepath, val_test_filepath)

    x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts, ngram_range=params.ngram_range,
                                     ngram_top_k=params.ngram_top_k,
                                     ngram_token_mode=params.ngram_token_mode,
                                     ngram_min_document_frequency=params.ngram_min_document_frequency)
    print()

    # TODO preprocess the data
