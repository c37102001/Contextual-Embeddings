import torch
from tqdm import tqdm
import ipdb


class Trainer:
    def __init__(self, model, optimizer, train_loader, valid_loader, total_epoch, ckpt_dir):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.total_epoch = total_epoch
        self.metric = AccuracyMetrics()
        self.device = torch.device('cuda:0')
        self.ckpt_dir = ckpt_dir

    def start(self):
        tqdm.write('[-] Start training!')
        bar = tqdm(range(self.total_epoch), desc='[Total progress]', leave=False)
        # self.load_ckpt(self.ckpt_dir / 'epoch-2.ckpt')
        for epoch in bar:
            self.metric.reset()
            self.epoch = epoch + 1
            self.run_epoch('train')
            self.run_epoch('eval')
            self.save_ckpt(self.epoch, self.ckpt_dir)

    def run_epoch(self, mode):
        if mode == 'train':
            dataloader = self.train_loader
            self.model.train()
            desc_prefix = 'Train'
        elif mode == 'eval':
            dataloader = self.valid_loader
            self.model.eval()
            desc_prefix = 'Eval'
        torch.set_grad_enabled(mode == 'train')

        bar = tqdm(dataloader, desc='[{} epoch {:2}]'.format(desc_prefix, self.epoch))
        for step, batch in enumerate(bar):
            if mode == 'train':
                loss = self.model(batch['context'], attention_mask=batch['mask'], labels=batch['label'])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                bar.set_postfix_str('loss={:.8f}'.format(loss))
            if mode == 'eval':
                prediction = self.model(batch['context'], attention_mask=batch['mask'])
                import ipdb
                if step < 5:
                    ipdb.set_trace()
                prediction = prediction.max(dim=1)[1]
                self.metric.update(prediction, batch['label'])
                bar.set_postfix_str('accuracy={:.8f}'.format(self.metric.value))
        bar.close()

    def save_ckpt(self, epoch, ckpt_dir):
        tqdm.write('[*] Saving model state')
        ckpt_path = ckpt_dir / 'epoch-{}.ckpt'.format(epoch)
        torch.save({
            'epoch': epoch,
            'net_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict()
        }, ckpt_path)
        tqdm.write('[-] Model state saved to {}\n'.format(ckpt_path))

    def load_ckpt(self, ckpt_path):
        print('[*] Loading model state')
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['net_state'])
        self.model.to(device=self.device)
        self.optimizer.load_state_dict(ckpt['optim_state'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self.device)


class AccuracyMetrics:
    def __init__(self):
        self.n = 0
        self.sum = 0

    def reset(self):
        self.sum = 0
        self.n = 0

    def update(self, prediction, target):
        prediction = prediction.detach()
        target = target.detach()
        self.n += len(prediction)
        self.sum += (prediction+1 == target).sum().item()

    @property
    def value(self):
        return self.sum / self.n

