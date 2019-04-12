import torch.nn.functional as F
import torch


class Metric:
    def __init__(self):
        self._set_name()
        self.reset()

    def _set_name(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def update(self, output, batch):
        raise NotImplementedError

    @property
    def value(self):
        raise NotImplementedError


class Accuracy(Metric):
    def __init__(self, device, key):
        self._device = device
        self._key = key
        super().__init__()

    def _set_name(self):
        self.name = 'acc({})'.format(self._key)

    def reset(self):
        self._sum = 0
        self._n = 0

    def update(self, output, batch):
        prediction = output[self._key].detach()             # (32, 590)
        target = batch[self._key].to(device=self._device)   # (32, 590)

        self._sum += (prediction == target).sum().item()    # 0
        self._n += len(prediction)                          # 32

    @property
    def value(self):
        return self._sum / self._n


class ELMoAccuracy(Metric):
    def __init__(self, device, key):
        self._device = device
        self._key = key
        super().__init__()

    def _set_name(self):
        self.name = 'acc({})'.format(self._key)

    def reset(self):
        self._sum = 0
        self._n = 0

    def update(self, output, batch):
        prediction = output[self._key].detach()             # [32, 590 * 2]
        target = batch[self._key]                           # [32, 590]
        rev_target_key = 'rev_' + self._key
        rev_target = batch[rev_target_key]
        target = torch.cat((target, rev_target), 1)        # [32, 590 * 2]

        prediction = prediction.view(-1).to(device=self._device)
        target = target.view(-1).to(device=self._device)

        self._sum += (prediction == target).sum().item()    # 0
        self._n += len(prediction)                          # 32

    @property
    def value(self):
        return self._sum / self._n
