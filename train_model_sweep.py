import argparse
import os
from pathlib import Path
import datetime
import wandb

from train_model import train_ngram_model
import utils
import load_and_process_data

parser = argparse.ArgumentParser()
parser.add_argument('--params_path', default='params.json',
                    help="Path to json file with parameters")

args = parser.parse_args()
assert os.path.isfile(args.params_path), "No json configuration file found at {}".format(args.params_path)
params = utils.Params(args.params_path)

train_filepath = os.path.join(params.data_dir, params.train_filename)
val_test_filepath = os.path.join(params.data_dir, params.val_test_filename)

train_texts, train_labels, val_texts, val_labels, _, _ = \
    load_and_process_data.load_train_val_test_datasets(train_filepath, val_test_filepath)

if params.preprocessing:
    train_texts = load_and_process_data.ingredients_preprocessing(train_texts)
    val_texts = load_and_process_data.ingredients_preprocessing(val_texts)

x_train, x_val = load_and_process_data.ngram_vectorize(train_texts, train_labels, val_texts)


# TODO implement logging instead of printing
def train_ngram_model_sweep_pass():

    sweep_version = 'sweep_v3'  # TODO change in both files (TODO make it a parameter)
    model_filename = "model_mlp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".h5"
    model_filepath = os.path.join(params.weights_dir, sweep_version, model_filename)
    Path(os.path.join(params.weights_dir, sweep_version)).mkdir(parents=True, exist_ok=True)

    use_wandb = True  # for debugging

    if use_wandb:
        wandb_run = wandb.init()

    trained_model, history, f1_micro_score = train_ngram_model(((train_texts, train_labels), (val_texts, val_labels)),
                                                               learning_rate=wandb.config.learning_rate,
                                                               epochs=wandb.config.epochs,
                                                               batch_size=wandb.config.batch_size,
                                                               layers=wandb.config.layers,
                                                               units=wandb.config.units,
                                                               dropout_rate=wandb.config.dropout_rate,
                                                               ngram_range=wandb.config.ngram_range,
                                                               ngram_top_k=wandb.config.ngram_top_k,
                                                               ngram_token_mode=wandb.config.ngram_token_mode,
                                                               ngram_min_document_frequency=wandb.config.ngram_min_document_frequency
                                                               )

    trained_model.save(model_filepath)

    if use_wandb:
        wandb.log({"f1_micro_score": f1_micro_score, "val_acc": history['val_acc'][-1]})
        wandb_run.finish()


if __name__ == '__main__':
    train_ngram_model_sweep_pass()

