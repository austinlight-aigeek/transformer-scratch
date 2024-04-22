import warnings
from training_loop import train_model


# Iterating through dataset to extract the original sentence and its translation
def get_all_sentences(ds, lang):
    for pair in ds:
        yield pair['translation'][lang]


warnings.filterwarnings('ignore')  # Filtering warnings
config = {
    'batch_size': 8,
    'num_epochs': 20,
    'lr': 10 ** -4,
    'seq_len': 350,
    'd_model': 512,
    'lang_src': 'en',
    'lang_tgt': 'it',
    'model_folder': 'weights',
    'model_basename': 'tmodel_',
    'preload': None,
    'tokenizer_file': 'tokenizer_{0}.json',
    'experiment_name': 'runs/tmodel'
}

train_model(config)  # Training model with the config arguments
