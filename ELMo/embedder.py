from common.utils import load_pkl
import torch
from ELMo.dataset import pad_to_len
from ELMo.elmo import ELMo
import sys
# sys.path.append('/nfs1/home/c37102001/Pycharm/PyTorch/adl-hw2/HW2/ELMo')
sys.path.append('/home/test/c37102001/HW2/ELMo')

class Embedder:

    def __init__(self, n_ctx_embs, ctx_emb_dim, embedding_path, net_cfg, ckpt_path):
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        self.embedding = load_pkl(embedding_path)
        self.word_dict = self.embedding.word_dict
        self.vectors = self.embedding.vectors

        self.device = torch.device('cuda:0')
        self.model = ELMo(self.device, self.vectors, **net_cfg)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['net_state'])
        self.model.to(device=self.device)

    def __call__(self, sentences, max_sent_len):

        # pad_len = min(max(map(len, sentences)), max_sent_len)
        # sentences = [['<bos>'] + sentence + ['<eos>'] for sentence in sentences]
        # sentences = [pad_to_len([self.embedding.to_index(word) for word in sentence],
        #                         pad_len, self.embedding.to_index('<pad>'))
        #              for sentence in sentences]
        # rev_sentences = [sentence[::-1] for sentence in sentences]
        # sentences = torch.tensor(sentences).to(self.device)
        # rev_sentences = torch.tensor(rev_sentences).to(self.device)
        #
        # elmo_embedding = self.model.get_elmo_embedding(sentences, rev_sentences)  # [32, 64, 512*2]
        #
        # return elmo_embedding.to('cpu').detach().numpy()

        pad_len = min(max(map(len, sentences)), max_sent_len) + 1
        sentences = [['<bos>'] + sentence + ['<eos>'] for sentence in sentences]

        context = [pad_to_len([self.embedding.to_index(word) for word in sentence[:-1]],
                              pad_len, self.embedding.to_index('<pad>'))
                   for sentence in sentences]
        rev_context = [pad_to_len([self.embedding.to_index(word) for word in sentence[:0:-1]],
                                  pad_len, self.embedding.to_index('<pad>'))
                       for sentence in sentences]

        context = torch.tensor(context).to(self.device)
        rev_context = torch.tensor(rev_context).to(self.device)
        elmo_embedding = self.model.get_elmo_embedding(context, rev_context)

        return elmo_embedding.to('cpu').detach().numpy()
