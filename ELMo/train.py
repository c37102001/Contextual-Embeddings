import argparse
import sys
import random
from pathlib import Path

import ipdb
import numpy as np
import torch
from box import Box
from torch.utils.data import Dataset, DataLoader

from common.base_model import BaseModel
from common.base_trainer import BaseTrainer
from common.losses import CrossEntropyLoss
from common.metrics import Accuracy
from common.utils import load_pkl
from ELMo.elmo import ELMo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Target model directory')  # ./model/
    args = parser.parse_args()

    return vars(args)


class Model(BaseModel):
    def _create_net_and_optim(self, voc_size, emb_size, net_cfg, optim_cfg):
        net = ELMo(voc_size, emb_size, **net_cfg)
        net.to(device=self._device)

        optim = getattr(torch.optim, optim_cfg.algo)
        optim = optim(
            filter(lambda p: p.requires_grad, net.parameters()), **optim_cfg.kwargs)

        return net, optim


class Trainer(BaseTrainer):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)

    def _run_batch(self, batch):        # batch:[32, 382]
        context_sent = self.embedding(batch['context'].to(self._device))  # [32, 382, 300]
        ipdb.set_trace()
        logits = self._model(context_sent).type(torch.FloatTensor)       # [32, 382, 45899]

        return {
            'predict': logits
        }


def main(model_dir):
    cfg = Box.from_yaml(filename=model_dir / 'config.yaml')

    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))  # 'cuda:0'
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_path = model_dir / 'log.csv'
    ckpt_dir = model_dir / 'ckpts'
    if any([p.exists() for p in [log_path, ckpt_dir]]):
        print('[!] Directory already contains saved ckpts/log')
        exit(1)
    ckpt_dir.mkdir()

    print('[*] Loading datasets from {}'.format(cfg.dataset_dir))
    dataset_dir = Path(cfg.dataset_dir)
    train_dataset = load_pkl(dataset_dir / 'train.pkl')
    valid_dataset = load_pkl(dataset_dir / 'valid.pkl')
    embedding = load_pkl(dataset_dir / 'embedding.pkl')
    embedding = embedding.vectors

    cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    train_data_loader = DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn, shuffle=True, **cfg.data_loader)
    valid_data_loader = DataLoader(
        valid_dataset, collate_fn=valid_dataset.collate_fn, **cfg.data_loader)

    voc_size = embedding.size(0)
    emb_size = embedding.size(1)
    model = Model(device, voc_size, emb_size, cfg.net, cfg.optim)

    trainer = Trainer(
        embedding, device,  cfg.train, train_data_loader, valid_data_loader, model,
        [CrossEntropyLoss(device, 'predict', 'labels')], [Accuracy(device, 'label')],
        log_path, ckpt_dir)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
