import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class ELMo(nn.Module):
    def __init__(self, voc_size, emb_size, d_model, dropout):
        super(ELMo, self).__init__()

        self.rnn = nn.LSTM(
            input_size=emb_size,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(d_model, voc_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)      # x: [32, 382, 300]  , r_out: [32, 382, 128])

        out = self.out(r_out)       # out: torch.Size([32, 382, 45899])
        return out
