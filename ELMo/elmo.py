import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class ELMo(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, project_size, dropout):
        super(ELMo, self).__init__()

        self.forward_rnn1 = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.backward_rnn1 = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.project1 = nn.Linear(hidden_size, project_size)

        self.forward_rnn2 = nn.LSTM(
            input_size=project_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.backward_rnn2 = nn.LSTM(
            input_size=project_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.project2 = nn.Linear(hidden_size, project_size)

        self.out = nn.Linear(project_size, voc_size)

    def forward(self, x, re_x):                     # x: [32, 382, 300]
        x, _ = self.forward_rnn1(x, None)           # x: [32, 382, 4096])
        x = self.project1(x)                        # x: [32, 382, 512]
        x, _ = self.forward_rnn2(x, None)           # x: [32, 382, 4096]
        x = self.project2(x)                        # x: [32, 382, 512]
        x = self.out(x)                             # x: [32, 382, 45899]

        re_x, _ = self.backward_rnn1(re_x, None)
        re_x = self.project1(re_x)
        re_x, _ = self.backward_rnn2(re_x, None)
        re_x = self.project2(re_x)
        re_x = self.out(re_x)                       # re_x: [32, 382, 45899]

        output = torch.cat((x, re_x), 1)            # [32, 382 * 2, 45899]

        return output
