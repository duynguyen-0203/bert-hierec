from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel


class NewsEncoder(ABC, RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, word_embed_dim: int, num_heads: int, query_dim: int,
                 self_attn_dropout: float, dropout: float):
        super().__init__(config=config)
        self.roberta = RobertaModel(config)
        self.reduce_dim = nn.Linear(in_features=config.hidden_size, out_features=word_embed_dim)
        self.word_embed_dropout = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=word_embed_dim, num_heads=num_heads, dropout=self_attn_dropout,
                                               batch_first=True)
        self.word_context_dropout = nn.Dropout(dropout)
        self.attention = AttentivePooling(input_dim=word_embed_dim, query_dim=query_dim)
        self._embed_dim = word_embed_dim

        self.init_weights()

    def forward(self, encoding: torch.tensor, attention_mask: torch.tensor):
        """
        Forward propagation
        :param encoding: shape [batch_size, seq_length]
        :param attention_mask: shape [batch_size, seq_length]
        :return: shape [batch_size, embed_dim]
        """
        word_embed = self.roberta(input_ids=encoding, attention_mask=attention_mask)[0]
        word_embed = self.reduce_dim(word_embed)
        word_embed = self.word_embed_dropout(word_embed)
        word_repr, _ = self.self_attn(query=word_embed, key=word_embed, value=word_embed,
                                   key_padding_mask=~attention_mask)
        word_repr = self.word_context_dropout(word_repr)
        news_repr = self.attention(embedding=word_repr, attention_mask=attention_mask)

        return news_repr

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim


class AttentivePooling(nn.Module):
    def __init__(self, input_dim: int, query_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=query_dim)
        self.linear2 = nn.Linear(in_features=query_dim, out_features=1)

    def forward(self, embedding: torch.tensor, attention_mask: torch.tensor):
        """
        Forward propagation
        :param embedding:
        :param attention_mask:
        :return:
        """
        attn_weight = self.linear1(embedding)
        attn_weight = torch.tanh(attn_weight)
        attn_weight = self.linear2(attn_weight).squeeze(dim=2)
        attn_weight.masked_fill_(~attention_mask, float('-inf'))
        attn_weight = torch_f.softmax(attn_weight, dim=1)
        attn_weight = torch.nan_to_num(attn_weight)
        seq_repr = torch.bmm(attn_weight.unsqueeze(dim=1), embedding).squeeze(dim=1)

        return seq_repr
