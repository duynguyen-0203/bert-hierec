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
        :param logits: shape [batch_size, npratio + 1]
        :param labels: one-hot vector [batch_size, npratio + 1]
        :return: Loss value
        """
        targets = labels.argmax(dim=1)
        loss = self._criterion(logits, targets)

        return loss

    @staticmethod
    def compute_eval_loss(logits: torch.tensor, labels: torch.tensor):
        """
        Compute loss for evaluation phase
        :param logits: shape [batch_size, 1]
        :param labels: shape [batch_size, 1] (binary value)
        :return: Float number
        """
        loss = -(torch_f.logsigmoid(logits) * labels).sum()

        return loss.item()
