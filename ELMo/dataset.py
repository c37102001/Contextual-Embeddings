import torch
from torch.utils.data import Dataset


class CorpusDataset(Dataset):
    """
    Args:
        data (list): List of samples(dict) with `context` and `labels` as index.
    """
    def __init__(self, data, padding=0, shuffle=False, max_pad_len=64):
        self.data = data
        self.padding = padding
        self.shuffle = shuffle
        self.max_pad_len = max_pad_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def collate_fn(self, datas):

        batch = dict()
        padded_len = min(self.max_pad_len, max([len(data) for data in datas]))

        batch['context'] = torch.tensor(
            [pad_to_len(data[:-1], padded_len, self.padding)
             for data in datas]
        )
        batch['label'] = torch.tensor(
            [pad_to_len(data[1:], padded_len, self.padding)
             for data in datas]
        )
        batch['rev_context'] = torch.tensor(
            [pad_to_len(data[:0:-1], padded_len, self.padding)
             for data in datas]
        )
        batch['rev_label'] = torch.tensor(
            [pad_to_len(data[-2::-1], padded_len, self.padding)
             for data in datas]
        )
        return batch


def pad_to_len(arr, padded_len, padding=0):
    if len(arr) >= padded_len:
        return arr[0:padded_len]
    else:
        pad_arr = [padding for _ in range(padded_len-len(arr))]
        return arr + pad_arr

