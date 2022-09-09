from abc import ABC, abstractmethod
import functools
import operator
import os
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from src.entities import Dataset


class BaseEvaluator(ABC):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.prob_predictions = []
        self.targets = []
        self.impression_ids = []
        self._convert_targets()

    @abstractmethod
    def _convert_targets(self):
        pass

    @abstractmethod
    def _convert_pred(self):
        pass

    @abstractmethod
    def eval_batch(self, logits: torch.tensor, impression_ids: torch.tensor):
        pass

    def _compute_scores(self, metrics: List[str], save_result: bool, path: str = None):
        self._convert_pred()
        assert len(self.targets) == len(self.prob_predictions)
        targets = flatten(self.targets)
        prob_predictions = flatten(self.prob_predictions)
        scores = {}

        for metric in metrics:
            if metric == 'auc':
                score = roc_auc_score(y_true=targets, y_score=prob_predictions)
                scores['auc'] = score
            elif metric == 'group_auc':
                list_score = [roc_auc_score(y_true=target, y_score=prob_prediction)
                              for target, prob_prediction in zip(targets, prob_predictions)]
                scores['group_auc'] = np.mean(list_score)
                if save_result:
                    save_scores(os.path.join(path, 'group_auc.txt'), list_score)
            elif metric == 'mrr':
                list_score = [compute_mrr_score(np.array(target), np.array(prob_prediction))
                              for target, prob_prediction in zip(targets, prob_predictions)]
                scores['mrr'] = np.mean(list_score)
                if save_result:
                    save_scores(os.path.join(path, 'mrr.txt'), list_score)
            elif metric.startswith('ndcg'):
                k = int(metric.split('@')[1])
                list_score = [compute_ndcg_score(y_true=np.array(target), y_score=np.array(prob_prediction), k=k)
                              for target, prob_prediction in zip(targets, prob_predictions)]
                scores[f'ndcg{k}'] = np.mean(list_score)
                if save_result:
                    save_scores(os.path.join(path, f'ndcg{k}.txt'), list_score)

        return scores


def compute_mrr_score(y_true: np.ndarray, y_score: np.ndarray):
    """
    Computing MRR score .
    :param y_true: Ground-truth labels
    :param y_score: Predicted score
    :return: MRR Score
    """
    rank = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, rank)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.sum(rr_score) / np.sum(y_true)


def compute_dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int):
    """
    Compute DCG@k score
    :param y_true:
    :param y_score:
    :param k:
    :return:
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)

    return np.sum(gains / discounts)


def compute_ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int):
    """
    Compute nDCG@k score
    :param y_true: Ground-truth labels
    :param y_score: Predicted score
    :param k:
    :return:
    """
    best = compute_dcg_score(y_true, y_true, k)
    actual = compute_dcg_score(y_true, y_score, k)

    return actual / best


def save_scores(path, scores):
    with open(path, mode='w', encoding='utf-8') as f:
        for score in scores:
            f.write(str(score))
            f.write('\n')


def flatten(lists: List[List]) -> List:
    return functools.reduce(operator.iconcat, lists, [])
