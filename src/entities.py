from collections import OrderedDict, Counter
from typing import List

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer


class News:
    def __init__(self, news_id: str, title: List[int], sapo: List[int], category: int):
        self._news_id = news_id
        self._title = title
        self._sapo = sapo
        self._category = category

    @property
    def news_id(self) -> int:
        return self._news_id

    @property
    def title(self) -> List[int]:
        return self._title

    @property
    def sapo(self) -> List[int]:
        return self._sapo

    @property
    def category(self) -> int:
        return self._category


class Impression:
    def __init__(self, impression_id: int, user_id: int, news: List[News], label: List[int]):
        self._impression_id = impression_id
        self._user_id = user_id
        self._news = news
        self._label = label

    @property
    def impression_id(self):
        return self._impression_id

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def news(self) -> List[News]:
        return self._news

    @property
    def label(self):
        return self._label


class Sample:
    def __init__(self, sample_id: int, user_id: int, clicked_news: List[News], impression: Impression):
        self._sample_id = sample_id
        self._user_id = user_id
        self._clicked_news = clicked_news
        self._impression = impression

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def clicked_news(self) -> List[News]:
        return self._clicked_news

    @property
    def impression(self) -> Impression:
        return self._impression


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, category2id: dict):
        super().__init__()
        self._name = data_name
        self._samples = OrderedDict()
        self._mode = Dataset.TRAIN_MODE
        self._tokenizer = tokenizer
        self._category2id = category2id

        self._news_id = 0
        self._id = 0

    def set_mode(self, mode: str):
        self._mode = mode

    def create_news(self, title: List[int], sapo: List[int], category: int) -> News:
        news = News(self._news_id, title, sapo, category)
        self._news_id += 1

        return news

    @staticmethod
    def create_impression(impression_id: int, user_id: int, news: List[News], label: List[int]) -> Impression:
        impression = Impression(impression_id, user_id, news, label)

        return impression

    def add_sample(self, user_id: int, clicked_news: List[News], impression: Impression):
        sample = Sample(self._id, user_id, clicked_news, impression)
        self._samples[self._id] = sample
        self._id += 1

    @property
    def samples(self) -> List[Sample]:
        return list(self._samples.values())

    @property
    def news_count(self) -> int:
        return self._news_id

    def __len__(self):
        if self._mode == Dataset.TRAIN_MODE:
            return len(self.samples)
        else:
            return len(set([sample.impression.impression_id for sample in self.samples]))

    def __getitem__(self, i: int):
        sample = self.samples[i]

        if self._mode == Dataset.TRAIN_MODE:
            return create_train_sample(sample, self._tokenizer, self._category2id)
        else:
            return create_eval_sample(sample, self._tokenizer, self._category2id)


def _create_sample(sample: Sample, tokenizer: PreTrainedTokenizer, category2id: dict) -> dict:
    num_category = len(category2id)
    # History click
    title_clicked_news_encoding = [news.title for news in sample.clicked_news]
    category_clicked_news_encoding = [news.category for news in sample.clicked_news]
    category_clicked_counter = dict(Counter(category_clicked_news_encoding))
    category_clicked_counter[category2id['pad']] = 0
    num_clicked = sum(list(category_clicked_counter.values()))
    history_category_ratio = [category_clicked_counter.get(key, 0) / num_clicked for key in list(category2id.values())]
    history_category_count_mask = [i != 0 for i in history_category_ratio]

    # Impression
    title_impression_encoding = [news.title for news in sample.impression.news]
    category_impression_encoding = [news.category for news in sample.impression.news]
    candidate_category_ratio = [history_category_ratio[i] for i in category_impression_encoding]

    # Create tensor
    impression_id = torch.tensor(sample.impression.impression_id)
    title_clicked_news_encoding = torch.tensor(title_clicked_news_encoding)
    title_impression_encoding = torch.tensor(title_impression_encoding)
    label = torch.tensor(sample.impression.label)
    candidate_category_ratio = torch.tensor(candidate_category_ratio)
    history_category_count_mask = torch.tensor(history_category_count_mask)
    # Create mask
    his_attn_mask = (title_clicked_news_encoding != tokenizer.pad_token_id)
    candidate_mask = (title_impression_encoding != tokenizer.pad_token_id)
    history_category_mask = torch.zeros(his_attn_mask.shape[0], num_category, dtype=bool)
    history_category_mask = history_category_mask.scatter_(
        1, torch.tensor(category_clicked_news_encoding).unsqueeze(dim=1), 1)
    history_category_mask = torch.transpose(history_category_mask, 0, 1)
    candidate_category_mask = torch.zeros(candidate_mask.shape[0], num_category, dtype=bool)
    candidate_category_mask = candidate_category_mask.scatter_(
        1, torch.tensor(category_impression_encoding).unsqueeze(dim=1), 1)

    return dict(impression_id=impression_id, candidate_encoding=title_impression_encoding,
                candidate_attn_mask=candidate_mask, candidate_category_mask=candidate_category_mask,
                history_encoding=title_clicked_news_encoding, history_attn_mask=his_attn_mask,
                history_category_mask=history_category_mask, candidate_category_ratio=candidate_category_ratio,
                history_category_count_mask=history_category_count_mask, label=label)


def create_train_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int) -> dict:
    return _create_sample(sample, tokenizer, num_category)


def create_eval_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int) -> dict:
    return _create_sample(sample, tokenizer, num_category)
