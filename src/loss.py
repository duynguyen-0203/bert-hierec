from abc import ABC, abstractmethod

import torch
import torch.nn.functional as torch_f


class AbstractLoss(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


class Loss(AbstractLoss):
    def __init__(self, criterion):
        self._criterion = criterion

    def compute(self, logits: torch.tensor, labels: torch.tensor):
        """
        Compute batch loss
        :param logits: shape [batch_size, K + 1]
        :param labels: one-hot vector [batch_size, K + 1]
        :return: Loss value
        """
        log_probs = torch_f.log_softmax(logits, dim=1)
        targets = labels.argmax(dim=1)
        loss = self._criterion(log_probs, targets)

        return loss
