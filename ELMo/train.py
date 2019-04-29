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
from common.utils import load_pkl
from ELMo.elmo import ELMo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, default='./ELMo/model/', help='Target model directory')
    args = parser.parse_args()

    return vars(args)


class Model(BaseModel):
    def _create_net_and_optim(self, embedding, net_cfg, optim_cfg):
        net = ELMo(self._device, embedding, **net_cfg)
        net.to(device=self._device)

        optim = getattr(torch.optim, optim_cfg.algo)
        optim = optim(filter(lambda p: p.requires_grad, net.parameters()), **optim_cfg.kwargs)

        return net, optim


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0

    def _run_batch(self, batch):
        loss = self._model(batch)
        self.step_count += 1
        if self.step_count % 20000 == 0:
            self._stat.log()
            self._save_step_ckpt()

        return {
            'loss': loss,
        }

    def _save_step_ckpt(self):
        self._model.save_state(self.step_count, self._stat.stat, self._ckpt_dir)


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

    model = Model(device, embedding, cfg.net, cfg.optim)

    # ckpt_path = ckpt_dir / 'epoch-60000.ckpt'
    # model.load_state(ckpt_path)

    trainer = Trainer(device,  cfg.train, train_data_loader, valid_data_loader, model,
                    [ELMoCrossEntropyLoss(device, 'loss', 'label', ignore_index=pad_idx, voc_size=embedding.size(0))],
        [], log_path, ckpt_dir)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
