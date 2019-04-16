from common.utils import load_pkl
import torch
from ELMo.dataset import pad_to_len
from ELMo.elmo import ELMo
import sys
# sys.path.append('/nfs1/home/c37102001/Pycharm/PyTorch/adl-hw2/HW2/ELMo')
sys.path.append('/home/test/c37102001/HW2/ELMo')

class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim, embedding_path, net_cfg, ckpt_path):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
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
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        pad_len = min(max(map(len, sentences)), max_sent_len)
        sentences = [['<bos>'] + sentence + ['<eos>'] for sentence in sentences]
        sentences = [pad_to_len([self.embedding.to_index(word) for word in sentence],
                                pad_len, self.embedding.to_index('<pad>'))
                     for sentence in sentences]
        rev_sentences = [sentence[::-1] for sentence in sentences]
        sentences = torch.tensor(sentences).to(self.device)
        rev_sentences = torch.tensor(rev_sentences).to(self.device)

        elmo_embedding = self.model.get_elmo_embedding(sentences, rev_sentences)  # [32, 64, 512*2]

        return elmo_embedding.to('cpu').detach().numpy()
