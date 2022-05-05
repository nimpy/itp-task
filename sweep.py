import wandb

from train_model_sweep import train_ngram_model_sweep_pass

if __name__ == '__main__':
    # TODO read sweep config from a file
    sweep_config = {
        'method': 'grid',
        'project': 'itp-task',
        'metric': {
            'goal': 'minimize',
            'name': 'f1_micro_score'
        },
        'parameters': {
            'layers': {
                'values': [1, 2, 3, 4]
            },
            'units': {
                'values': [32, 64, 128]
            },
            'dropout_rate': {
                'values': [0.2, 0.35, 0.5]
            },
            'epochs': {
                'values': [150]
            },
            'batch_size': {
                'values': [128]
            },
            'learning_rate': {
                'values': [0.001, 0.0001, 0.00001]
            },
            'ngram_range': {
                'values': [2]
            },
            'ngram_top_k': {
                'values': [20000]
            },
            'ngram_token_mode': {
                'values': ["word"]
            },
            'ngram_min_document_frequency': {
                'values': [2]
            }
        }
    }

    sweep_version = 'sweep_v3'  # TODO change in both files (TODO make it a parameter)

    sweep_id = wandb.sweep(sweep_config, project="itp-task")
    wandb.agent(sweep_id, function=train_ngram_model_sweep_pass)
