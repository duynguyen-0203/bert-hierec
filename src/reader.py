import csv
import random
from typing import List, Tuple

from transformers import PreTrainedTokenizer

from src.entities import Dataset, News


class Reader:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_title_length: int, max_sapo_length: int, user2id: dict,
                 category2id: dict, max_his_click: int):
        self._tokenizer = tokenizer
        self._max_title_length = max_title_length
        self._max_sapo_length = max_sapo_length
        self._user2id = user2id
        self._category2id = category2id
        self._max_his_click = max_his_click

    def read_train_dataset(self, data_name: str, news_path: str, behaviors_path: str) -> Dataset:
        dataset, news_dataset = self._read(data_name, news_path)
        with open(behaviors_path, mode='r', encoding='utf-8', newline='') as f:
            behaviors_tsv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(behaviors_tsv):
                self._parse_train_line(line, news_dataset, dataset)

        return dataset

    def read_eval_dataset(self, data_name: str, news_path: str, behaviors_path: str) -> Dataset:
        dataset, news_dataset = self._read(data_name, news_path)
        with open(behaviors_path, mode='r', encoding='utf-8', newline='') as f:
            behaviors_tsv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(behaviors_tsv):
                self._parse_eval_line(line, news_dataset, dataset)

        return dataset

    def _read(self, data_name: str, news_path: str) -> Tuple[Dataset, dict]:
        dataset = Dataset(data_name, self._tokenizer, self._category2id['pad'], self._max_his_click)
        news_dataset = self._read_news_info(news_path, dataset)

        return dataset, news_dataset

    def _read_news_info(self, news_path, dataset: Dataset) -> dict:
        pad_news_obj = dataset.create_news([self._tokenizer.pad_token_id] * self._max_title_length,
                                           [self._tokenizer.pad_token_id] * self._max_sapo_length,
                                           self._category2id['pad'])
        news_dataset = {'pad': pad_news_obj}
        with open(news_path, mode='r', encoding='utf-8', newline='') as f:
            news_tsv = csv.reader(f, delimiter='\t')
            for line in news_tsv:
                title_encoding = self._tokenizer.encode(line[1], add_special_tokens=True, padding='max_length',
                                                        truncation=True, max_length=self._max_title_length)
                category_id = self._category2id.get(line[2], self._category2id['unk'])
                sapo_encoding = self._tokenizer.encode(line[3], add_special_tokens=True, padding='max_length',
                                                       truncation=True, max_length=self._max_sapo_length)
                news = dataset.create_news(title_encoding, sapo_encoding, category_id)
                news_dataset[line[0]] = news

        return news_dataset

    def _parse_train_line(self, line, news_dataset, dataset):
        user_id = self._user2id.get(line[1], self._user2id['unk'])
        history_clicked = [news_dataset[news_id] for news_id in line[3].split()]
        pos_news = [news_dataset[news_id] for behavior in line[4].split() for news_id, label in behavior.split('-')
                    if label == 1]
        neg_news = [news_dataset[news_id] for behavior in line[4].split() for news_id, label in behavior.split('-')
                    if label == 0]

    def _parse_eval_line(self, line, news_dataset, dataset):
        pass


def sample_news(list_news: List[News], num_news: int, pad: News):
    if len(list_news) >= num_news:
        return random.sample(list_news, k=num_news)
    else:
        return list_news + [pad] * (num_news - len(list_news))
