import argparse
import sys
from pathlib import Path

import ipdb
import numpy as np
import torch


from common.base_model import BaseModel
from common.base_trainer import BaseTrainer
from common.losses import CrossEntropyLoss
from common.metrics import Accuracy
from common.utils import load_pkl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Target model directory')  # model/MODEL_NAME
    args = parser.parse_args()

    return vars(args)


class Model(BaseModel):
    def _create_net_and_optim(self, word_vocab, char_vocab, net_cfg, optim_cfg):
        net = BCN(word_vocab, char_vocab, **net_cfg)
        net.to(device=self._device)

        optim = getattr(torch.optim, optim_cfg.algo)
        optim = optim(
            filter(lambda p: p.requires_grad, net.parameters()), **optim_cfg.kwargs)

        return net, optim


class Trainer(BaseTrainer):
    def __init__(self, max_sent_len, elmo_embedder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_sent_len = max_sent_len
        self._elmo_embedder = elmo_embedder

    def _run_batch(self, batch):
        text_word = batch['text_word'].to(device=self._device)      # (32_batch_size, 40_text_words)
        text_char = batch['text_char'].to(device=self._device)      # (32, 40, 16_letters)

        text_pad_mask = batch['text_pad_mask'].to(device=self._device)

        logits = self._model(text_word, text_char, text_ctx_emb, text_pad_mask)

        label = logits.max(dim=1)[1]
        return {
            'logits': logits,
            'label': label
        }


def main(model_dir):
    cfg = None
    dataset_dir = None

    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))  # 'cuda:0'
    word_vocab = load_pkl(dataset_dir / 'word.pkl')
    char_vocab = load_pkl(dataset_dir / 'char.pkl')
    model = Model(device, word_vocab, char_vocab, cfg.net, cfg.optim)

    trainer = Trainer(
        cfg.data_loader.max_sent_len, elmo_embedder, device, cfg.train,
        train_data_loader, dev_data_loader, model,
        [CrossEntropyLoss(device, 'logits', 'label')], [Accuracy(device, 'label')],
        log_path, ckpt_dir)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
