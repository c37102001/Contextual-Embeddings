import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


class BertDataset(Dataset):
    """
    Args:
        data (list): List of samples(dict) with `text` and `label` as index.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        batch = dict()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_contexts = [tokenizer.tokenize(data['context']) for data in datas]
        indexed_contexts = [tokenizer.convert_tokens_to_ids(text) for text in tokenized_contexts]
        max_pad_len = max(map(len, indexed_contexts))

        masks = [[1 for _ in text] for text in indexed_contexts]
        batch['context'] = torch.LongTensor([pad_to_len(text, max_pad_len) for text in indexed_contexts]).to('cuda')
        batch['mask'] = torch.LongTensor([pad_to_len(mask, max_pad_len) for mask in masks]).to('cuda')
        batch['label'] = torch.LongTensor([int(data['label'])-1 for data in datas]).to('cuda')

        return batch


def pad_to_len(arr, padded_len, padding=0):
    if len(arr) >= padded_len:
        return arr[0:padded_len]
    else:
        pad_arr = [padding for _ in range(padded_len-len(arr))]
        return arr + pad_arr

