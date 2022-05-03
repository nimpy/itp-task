from tensorflow.python.keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import collections
import argparse
import os

import load_and_process_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--params_path', default='params.json',
                    help="Path to json file with parameters")
parser.add_argument('--model_filename', default='model_mlp_20220502_195902.h5',
                    help="Filename of the model to be loaded")

args = parser.parse_args()
assert os.path.isfile(args.params_path), "No json configuration file found at {}".format(args.params_path)
params = utils.Params(args.params_path)

# loading the datasets here, in order not to do it every time evaluation is called from another module
train_filepath = os.path.join(params.data_dir, params.train_filename)
val_test_filepath = os.path.join(params.data_dir, params.val_test_filename)

train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
    load_and_process_data.load_train_val_test_datasets(train_filepath, val_test_filepath)

if params.preprocessing:
    train_texts = load_and_process_data.ingredients_preprocessing(train_texts)
    val_texts = load_and_process_data.ingredients_preprocessing(val_texts)
    test_texts = load_and_process_data.ingredients_preprocessing(test_texts)

x_train, x_test = load_and_process_data.ngram_vectorize(train_texts, train_labels, test_texts)
_, x_val = load_and_process_data.ngram_vectorize(train_texts, train_labels, val_texts)


# We'll use F1 score (micro) as our single-number evaluation metric, and make decisions based on this metric.
# The reasoning: F1 score is the most common evaluation metric for classification, i.e. the 'vanilla' evaluation
# metric. If, later on, it would turn out that there is a metric that better incorporates what we want to achieve,
# then we could use that one.
# Regarding the choice of *micro* F1 score, it takes into account the class frequency, which is a desirable property
# in this case, since no cuisine is more important than the others (it wouldn't be a desirable property in the case
# of, e.g. classification of different types of tumors, where it is more important to discover some types of tumors
# than other types). Furthermore, the differences between micro F1, macro F1 and weighted F1 scores on this dataset
# are very small (<0.005), so this choice will not impact the evaluation greatly.
def calculate_evaluation_metric(model, x, y_true):
    """Evaluates the model based on the micro F1 score.
    # Arguments
        model: a trained neural network.
        x: list, inputs to the model, in our case the lists of ingredients.
        y_true: np.ndarray, ground truth labels for x.
    """
    # TODO this method could also have the evaluation metric function as an argument
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='micro')


def evaluate_model_on_test_set(model):
    return calculate_evaluation_metric(model, x_test, test_labels)


if __name__ == '__main__':

    example_model_filepath = os.path.join(params.weights_dir, args.model_filename)
    example_model = load_model(example_model_filepath)

    y_test_pred = example_model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    print("--- TEST SET ---")
    print(collections.Counter((y_test_pred == test_labels)))
    print(confusion_matrix(test_labels, y_test_pred))
    print(f1_score(test_labels, y_test_pred, average='micro'))
    print(f1_score(test_labels, y_test_pred, average='macro'))
    print(f1_score(test_labels, y_test_pred, average='weighted'))

    y_val_pred = example_model.predict(x_val)
    y_val_pred = np.argmax(y_val_pred, axis=1)
    print("\n--- VAL SET ---")
    print(collections.Counter((y_val_pred == val_labels)))
    print(confusion_matrix(val_labels, y_val_pred))
    print(f1_score(val_labels, y_val_pred, average='micro'))
    print(f1_score(val_labels, y_val_pred, average='macro'))
    print(f1_score(val_labels, y_val_pred, average='weighted'))

    y_train_pred = example_model.predict(x_train)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    print("\n--- TRAIN SET ---")
    print(collections.Counter((y_train_pred == train_labels)))
    print(confusion_matrix(train_labels, y_train_pred))
    print(f1_score(train_labels, y_train_pred, average='micro'))
    print(f1_score(train_labels, y_train_pred, average='macro'))
    print(f1_score(train_labels, y_train_pred, average='weighted'))

    print("\nMicro F1 score on the test set:", evaluate_model_on_test_set(example_model))
