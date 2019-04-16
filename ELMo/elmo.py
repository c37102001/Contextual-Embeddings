import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class ELMo(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, project_size, dropout):
        super(ELMo, self).__init__()

        self.emb_linear = nn.Linear(emb_size, project_size)

        self.forward_rnn1 = nn.LSTM(
            input_size=project_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.forward_project1 = nn.Linear(hidden_size, project_size)
        self.forward_rnn2 = nn.LSTM(
            input_size=project_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.forward_project2 = nn.Linear(hidden_size, project_size)

        self.backward_rnn1 = nn.LSTM(
            input_size=project_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.backward_project1 = nn.Linear(hidden_size, project_size)
        self.backward_rnn2 = nn.LSTM(
            input_size=project_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.backward_project2 = nn.Linear(hidden_size, project_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.AdaptiveLogSoftmaxWithLoss(project_size, voc_size, [100, 1000, 10000])

        # self.out = nn.Linear(project_size, voc_size)

    def forward(self, x, re_x, x_label, re_x_label):        # x: [32, 64, 300]
        x = self.emb_linear(x)                              # x: [32, 64, 512]
        x, _ = self.forward_rnn1(x, None)                   # x: [32, 64, 4096]
        x = self.dropout(x)
        x = self.forward_project1(x)                        # x: [32, 64, 512]
        x, _ = self.forward_rnn2(x, None)                   # x: [32, 64, 4096]
        x = self.dropout(x)
        x = self.forward_project2(x)                        # x: [32, 64, 512]
        x = x.view(x.size(0) * x.size(1), -1)               # x: [32*64 , 512]
        x_loss = self.out(x, x_label.view(-1)).loss         # x_label: [32*64]
        x_predict = self.out.predict(x)     # tensor([ 8,  6,  6, 16, 14, 16, 16,  9,  4,  7,  5,  7,  8, 14,  3])

        re_x = self.emb_linear(re_x)
        re_x, _ = self.backward_rnn1(re_x, None)
        re_x = self.dropout(re_x)
        re_x = self.backward_project1(re_x)
        re_x, _ = self.backward_rnn2(re_x, None)
        re_x = self.dropout(re_x)
        re_x = self.backward_project2(re_x)
        re_x = re_x.view(re_x.size(0) * re_x.size(1), -1)
        re_x_loss = self.out(re_x, re_x_label.view(-1)).loss
        re_x_predict = self.out.predict(re_x)

        loss = (x_loss + re_x_loss) / 2.0
        predict = x_predict + re_x_predict

        return loss, predict

    def get_elmo_embedding(self, x, re_x):

        x = self.emb_linear(x)
        re_x = self.emb_linear(re_x)
        e = torch.cat((x, re_x), 2)             # [32, 64, 512*2]

        x, _ = self.forward_rnn1(x, None)       # x: [32, 64, 4096]
        x = self.forward_project1(x)            # x: [32, 64, 512]
        re_x, _ = self.backward_rnn1(re_x, None)
        re_x = self.backward_project1(re_x)
        o1 = torch.cat((x, re_x), 2)            # [32, 64, 512*2]

        x, _ = self.forward_rnn2(x, None)
        x = self.forward_project2(x)
        re_x, _ = self.backward_rnn2(re_x, None)
        re_x = self.backward_project2(re_x)
        o2 = torch.cat((x, re_x), 2)

        return e, o1, o2
