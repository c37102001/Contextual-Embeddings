import torch
import torch.nn as nn
import torch.nn.functional as F


class Elmo(nn.Module):
    def __init__(self, words, ctx_emb_dim):
        super(Elmo, self).__init__()

        self.embedding = nn.Embedding(len(words), ctx_emb_dim)
