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
from common.losses import ELMoCrossEntropyLoss
from common.metrics import ELMoAccuracy
from common.utils import load_pkl
from ELMo.elmo import ELMo
import tqdm
from datetime import datetime
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, default='./ELMo/model/', help='Target model directory')
    args = parser.parse_args()

    return vars(args)


class Model(BaseModel):
    def _create_net_and_optim(self, voc_size, emb_size, net_cfg, optim_cfg):
        net = ELMo(voc_size, emb_size, **net_cfg)
        net.to(device=self._device)

        optim = getattr(torch.optim, optim_cfg.algo)
        optim = optim(filter(lambda p: p.requires_grad, net.parameters()), **optim_cfg.kwargs)

        return net, optim


class Trainer(BaseTrainer):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding = self.embedding.to(self._device)

    def _run_batch(self, batch):                                                    # batch:[32, 64]
        context = self.embedding(batch['context'].to(self._device))                 # [32, 64, 300]
        rev_context = self.embedding(batch['rev_context'].to(self._device))
        context_label = batch['label'].to(self._device)                             # [32, 64]
        rev_context_label = batch['rev_label'].to(self._device)

        loss, predict = self._model(context, rev_context, context_label, rev_context_label)

        return {
            'loss': loss,
            'label': predict
        }

    def _save_ckpt(self):
        self._model.save_state(self._epoch, self._stat.stat, self._ckpt_dir)
        embedding_path = self._ckpt_dir / 'embedding-{}.pkl'.format(self._epoch)
        with open(embedding_path, 'wb') as f:
            pickle.dump(self.embedding, f)


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

    pad_idx = embedding.to_index('<pad>')
    embedding = embedding.vectors

    cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    train_data_loader = DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn, shuffle=train_dataset.shuffle, **cfg.data_loader)
    valid_data_loader = DataLoader(
        valid_dataset, collate_fn=valid_dataset.collate_fn, shuffle=valid_dataset.shuffle, **cfg.data_loader)

    voc_size = embedding.size(0)
    emb_size = embedding.size(1)
    model = Model(device, voc_size, emb_size, cfg.net, cfg.optim)

    trainer = Trainer(
        embedding, device,  cfg.train, train_data_loader, valid_data_loader, model,
        [ELMoCrossEntropyLoss(device, 'loss', 'label', ignore_index=pad_idx, voc_size=voc_size)],
        [ELMoAccuracy(device, 'label')], log_path, ckpt_dir)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
