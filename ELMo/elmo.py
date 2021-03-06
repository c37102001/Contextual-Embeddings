import torch
import torch.nn as nn
import ipdb


class ELMo(nn.Module):
    def __init__(self, device, embedding, hidden_size, project_size, dropout):
        super(ELMo, self).__init__()
        self.device = device
        voc_size = embedding.size(0)
        emb_size = embedding.size(1)

        self.embedding = torch.nn.Embedding(voc_size, emb_size)
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding = self.embedding.to(self.device)

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

    def forward(self, batch):

        x = self.embedding(batch['context'].to(self.device))  # [32, 64, 300]
        re_x = self.embedding(batch['rev_context'].to(self.device))
        x_label = batch['label'].to(self.device)            # [32, 64]
        re_x_label = batch['rev_label'].to(self.device)
        mask = batch['mask'].to(self.device)
        mask_len = [i.item() if not i == 0 else 1.0 for i in batch['mask'].sum(dim=1)]
        mask_len = torch.FloatTensor(mask_len).to(self.device)  # [32] = [63, 62, 48, ...]
        batch_size = x.size()[0]
        seq_len = x.size()[1]

        x = self.emb_linear(x)                              # x: [32, 64, 512]
        x, _ = self.forward_rnn1(x, None)                   # x: [32, 64, 4096]
        # x = self.dropout(x)
        x = self.forward_project1(x)                        # x: [32, 64, 512]
        # x = self.dropout(x)
        x, _ = self.forward_rnn2(x, None)                   # x: [32, 64, 4096]
        # x = self.dropout(x)
        x = self.forward_project2(x)                        # x: [32, 64, 512]
        x = x.view(x.size(0) * x.size(1), -1)               # x: [32*64 , 512]
        # x_loss = self.out(x, x_label.view(-1)).loss         # x_label: [32*64]

        x_loss = self.out(x, x_label.view(-1)).output.view(batch_size, seq_len)         # x_label: [32*64]
        x_loss = x_loss*mask  # [32,64]
        x_loss = x_loss.sum(dim=1)  # [32]
        x_loss = (x_loss / mask_len).mean()

        re_x = self.emb_linear(re_x)
        re_x, _ = self.backward_rnn1(re_x, None)
        # re_x = self.dropout(re_x)
        re_x = self.backward_project1(re_x)
        # re_x = self.dropout(re_x)
        re_x, _ = self.backward_rnn2(re_x, None)
        # re_x = self.dropout(re_x)
        re_x = self.backward_project2(re_x)
        re_x = re_x.view(re_x.size(0) * re_x.size(1), -1)
        # re_x_loss = self.out(re_x, re_x_label.view(-1)).loss

        re_x_loss = self.out(re_x, re_x_label.view(-1)).output.view(batch_size, seq_len)  # x_label: [32*64]
        re_x_loss = re_x_loss * mask  # [32,64]
        re_x_loss = re_x_loss.sum(dim=1)  # [32]
        re_x_loss = (re_x_loss / mask_len).mean()

        loss = (x_loss + re_x_loss) / (-2.0)
        return loss

    def get_elmo_embedding(self, sentences, rev_sentences):

        x = self.embedding(sentences)  # [32, 64, 300]
        re_x = self.embedding(rev_sentences)

        x = self.emb_linear(x)
        re_x = self.emb_linear(re_x)
        e = torch.cat((x[:, 1:, :], re_x[:, 1:, :]), 2)             # [32, 64, 512*2]

        x, _ = self.forward_rnn1(x, None)       # x: [32, 64, 4096]
        x = self.forward_project1(x)            # x: [32, 64, 512]
        re_x, _ = self.backward_rnn1(re_x, None)
        re_x = self.backward_project1(re_x)
        o1 = torch.cat((x[:, 1:, :], re_x[:, 1:, :]), 2)            # [32, 64, 512*2]

        x, _ = self.forward_rnn2(x, None)
        x = self.forward_project2(x)
        re_x, _ = self.backward_rnn2(re_x, None)
        re_x = self.backward_project2(re_x)
        o2 = torch.cat((x[:, 1:, :], re_x[:, 1:, :]), 2)

        elmo_embedding = torch.cat((e.unsqueeze(2), o1.unsqueeze(2), o2.unsqueeze(2)), 2)  # [32, 64, 3, 1024]

        return elmo_embedding
