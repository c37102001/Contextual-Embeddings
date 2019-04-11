import torch
from elmo import ELMo
from tqdm import tqdm


class Predictor:
    def __init__(self, embedding, batch_size=32, max_epochs=10, valid=None, device=None,
                 learning_rate=5e-4, max_iters_in_epoch=1e20):

        self.model = ELMo(embedding.size(1))
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_iters_in_epoch = max_iters_in_epoch
        self.valid = valid
        self.learning_rate = learning_rate
        self.epoch = 0

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if not torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)

    def fit_dataset(self, train_dataloader, valid_dataloader):
        while self.epoch < self.max_epochs:
            self.run_epoch(train_dataloader, training=True)
            self.run_epoch(valid_dataloader, training=False)
            self.epoch += 1

    def run_epoch(self, dataloader, training):
        loss = 0
        if training:
            iter_in_epoch = min(len(dataloader), self.max_iters_in_epoch)
            description = 'training'

        else:
            iter_in_epoch = len(dataloader)
            description = 'evaluating'

        trange = tqdm(enumerate(dataloader), total=iter_in_epoch, desc=description)

        for i, batch in trange:
            if training and i >= iter_in_epoch:
                break

            if training:
                batch_loss = self.run_iter(batch)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            else:
                with torch.no_grad():
                    batch_loss = self.run_iter(batch)

            # accumulate loss and metric scores
            loss += batch_loss.item()

            trange.set_postfix(loss=loss / (i + 1))

    def run_iter(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        labels = self.embedding(batch['labels'].to(self.device))

        outputs = self.model.forward(context)
        loss = self.loss(outputs.to(self.device), labels.float().to(self.device))
        return loss
