# In The Pocket – ML Engineer Test Case – Nina Zizakic

### Structure of the repository

📁 [notebooks/](notebooks) – a directory with Jupyter notebooks (currently has only one)
* 📄 [playing_with_data.ipynb](notebooks/playing_with_data.ipynb) – notebook where I initially explored the data

📄 [build_model.py](build_model.py) – contains the architecture of the neural network used for this project (multi-layer perceptron)

📄 [evaluate_model.py](evaluate_model.py) – contains the implementations of model evaluation based on different metrics

📄 [load_and_process_data.py](load_and_process_data.py) – a file with the implementation of data loading, text preprocessing and feature extraction

📄 [params.json](params.json) – contains parameters for training (e.g. learning rate) and some config parameters (e.g. path to the data directory). (It should probably be separated into two json files: params.json and config.json.) 

📄 [sweep.py](sweep.py) – the script for performing Weights & Biases hyperparameter sweep 

📄 [train_model.py](train_model.py) – a file with model training implementation

📄 [train_model_sweep.py](train_model_sweep.py) – a file for the Weights & Biases hyperparameter sweep, which is called from sweep.py for every set of hyperparameters 

📄 [utils.py](utils.py) – a file with miscellaneous utilities, currently has only implementation of loading the params dictionary from the json file 


This structure of the repository is initial, and I would update it and improve it further on as I would see fit.


#### Suggested order in which to look at the files:
0. This readme file :)
1. [playing_with_data.ipynb](notebooks/playing_with_data.ipynb)
2. [load_and_process_data.py](load_and_process_data.py)
3. [build_model.py](build_model.py)
4. [train_model.py](train_model.py)
5. [params.json](params.json)
6. [evaluate_model.py](evaluate_model.py)
7. [sweep.py](sweep.py)
8. [train_model_sweep.py](train_model_sweep.py)
9. [utils.py](utils.py)

