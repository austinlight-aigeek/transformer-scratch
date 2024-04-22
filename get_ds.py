from tokenizer import build_tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from bilingual_dataset import BilingualDataset


def get_dataset(config):
    # Loading the train portion of the OpusBooks dataset.
    # The Language pairs will be defined in the 'config' dictionary we will build later
    dataset_row = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Building or loading tokenizer for both the source and target languages
    tokenizer_src = build_tokenizer(config, dataset_row, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, dataset_row, config['lang_tgt'])

    # Splitting the dataset for training and validation
    train_ds_size = int(0.9 * len(dataset_row))  # 90% for training
    val_ds_size = len(dataset_row) - train_ds_size  # 10% for validation
    train_ds_raw, val_ds_raw = random_split(dataset_row, [train_ds_size, val_ds_size])  # Randomly splitting the dataset

    # Processing data with the BilingualDataset class, which we will define below
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                              config['seq_len'])

    # Iterating over the entire dataset and printing the maximum length found in the sentences of both the source and
    # target languages
    max_len_src = 0
    max_len_tgt = 0
    for pair in dataset_row:
        src_ids = tokenizer_src.encode(pair['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(pair['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Creating dataloaders for the training and validation sets
    # Dataloaders are used to iterate over the dataset in batches during training and validation
    # Batch size will be defined in the config dictionary
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # Returning the DataLoader objects and tokenizers
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
