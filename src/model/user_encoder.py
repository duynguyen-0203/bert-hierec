import torch
import torch.nn as nn
import torch.nn.functional as torch_f

from src.model.news_encoder import NewsEncoder


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, num_category: int, category_embed_dim: int, his_length: int,
                 num_click_embed_dim: int):
        super().__init__()
        self.news_encoder = news_encoder
        self.category_embedding = nn.Parameter(data=nn.init.constant_(torch.empty(num_category, category_embed_dim),
                                                                      val=0.0),
                                               requires_grad=True)
        self.init_news_attn_weight = nn.Linear(in_features=news_encoder.embedding_dim, out_features=1, bias=False)
        self.init_category_attn_weight = nn.Linear(in_features=news_encoder.embedding_dim, out_features=1, bias=False)
        self.num_click_embedding_layer = nn.Embedding(num_embeddings=his_length + 1, embedding_dim=num_click_embed_dim)
        self.num_click_scorer = nn.Linear(in_features=num_click_embed_dim, out_features=1)

    def forward(self, encoding: torch.tensor, attention_mask: torch.tensor, category_mask: torch.tensor):
        """
        Forward propagation
        :param encoding: shape [batch_size, his_length, seq_length]
        :param attention_mask: shape [batch_size, his_length, seq_length]
        :param category_mask: shape [batch_size, num_category, his_length]
        :return:
        """
        batch_size = encoding.shape[0]
        his_length = encoding.shape[1]
        num_category = category_mask.shape[0]

        # News representation
        encoding = encoding.view(batch_size * his_length, -1)
        attention_mask = attention_mask.view(batch_size * his_length, -1)
        news_repr = self.news_encoder(encoding=encoding, attention_mask=attention_mask)
        news_repr = news_repr.view(batch_size, his_length, -1)

        # Topic-level interest representation
        news_attn_weight = self.init_news_attn_weight(news_repr).squeeze(dim=2)
        news_attn_weight = news_attn_weight.unsqueeze(dim=1).expand(news_attn_weight.shape[0], num_category, -1)
        news_attn_weight.masked_fill_(~category_mask, float('-inf'))
        news_attn_weight = torch_f.softmax(news_attn_weight)
        news_attn_weight = torch.nan_to_num(news_attn_weight)

        category_interest_repr = torch.bmm(news_attn_weight, news_repr)
        category_repr = category_interest_repr + self.category_embedding

        # User-level interest representation
        category_attn_weight = self.init_category_attn_weight(category_repr).squeeze(dim=2)
        category_count = category_mask.long().sum(-1, keepdims=False)
        num_click_embedding = self.num_click_embedding_layer(category_count)
        num_click_score = self.num_click_scorer(num_click_embedding).squeeze(dim=2)
        final_attn_weight = category_attn_weight + num_click_score
        category_count_mask = (category_count == 0)
        final_attn_weight.masked_fill_(category_count_mask, float('-inf'))
        final_attn_weight = torch_f.softmax(final_attn_weight)
        user_repr = torch.bmm(final_attn_weight.unsqueeze(dim=1), category_repr).squeeze(dim=1)

        return category_repr, user_repr
























