import argparse
import csv
import sys
from pathlib import Path
import ipdb
import torch
from torch.utils.data import DataLoader

from dataset import BertDataset
from trainer import Trainer
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, BertAdam


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_dir', type=Path, help='csv_dir')  # data/classification
    args = parser.parse_args()

    return vars(args)


def load_csv_data(csv_path):
    dataset = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            data = {
                'context': '[CLS] ' + r['text'],
                'label': r['label']
            }
            dataset.append(data)
    return BertDataset(dataset)


def main(csv_dir):
    EPOCH = 10
    BATCH_SIZE = 32

    csv_dir = Path(csv_dir)
    train_csv_path = csv_dir / 'train.csv'
    dev_csv_path = csv_dir / 'dev.csv'
    ckpt_dir = Path('./Bert/ckpts')

    train_data = load_csv_data(train_csv_path)
    valid_data = load_csv_data(dev_csv_path)

    train_loader = DataLoader(dataset=train_data, collate_fn=train_data.collate_fn,
                              batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data,  collate_fn=valid_data.collate_fn,
                              batch_size=BATCH_SIZE, shuffle=False)

    # config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    #                     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    #
    # bert_model = BertForSequenceClassification(config, num_labels=5).to('cuda')

    # optimizer = getattr(torch.optim, 'Adam')
    # optimizer = optimizer(
    #     filter(lambda p: p.requires_grad, bert_model.parameters()), lr=1.0e-3, weight_decay=1.0e-6)

    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5).to('cuda')

    w = torch.tensor([0.1, -0.2, -0.1], requires_grad=True)
    optimizer = BertAdam(params=[w], lr=2e-2, weight_decay=1.0e-6, max_grad_norm=-1)

    trainer = Trainer(bert_model, optimizer, train_loader, valid_loader, EPOCH, ckpt_dir)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
