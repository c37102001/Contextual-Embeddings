import torch
from torch.utils.data import Dataset


class CorpusDataset(Dataset):
    """
    Args:
        data (list): List of samples(dict) with `context` and `labels` as index.
    """
    def __init__(self, data, padding=0, padded_len=300, shuffle=True):
        self.data = data
        self.padded_len = padded_len
        self.padding = padding
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        if len(data['context']) > self.padded_len:
            data['context'] = data['contexts'][:self.padded_len]
        if len(data['labels']) > self.padded_len:
            data['labels'] = data['labels'][:self.padded_len]
        return data

    def collate_fn(self, datas):

        batch = dict()
        padded_len = min(self.padded_len, max([len(data['context']) for data in datas]))

        batch['context'] = torch.tensor(
            [pad_to_len(data['context'], padded_len, self.padding)
             for data in datas]
        )
        batch['labels'] = torch.tensor(
            [pad_to_len(data['labels'], padded_len, self.padding)
             for data in datas]
        )
        return batch


def pad_to_len(arr, padded_len, padding=0):
    if len(arr) >= padded_len:
        return arr[0:padded_len]
    else:
        pad_arr = [padding for _ in range(padded_len-len(arr))]
        return arr + pad_arr

