import torch
import torch.nn as nn

from src.model.news_encoder import NewsEncoder
from src.model.user_encoder import UserEncoder


class HieRec(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, user_encoder: UserEncoder, dropout: float):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, candidate_encoding: torch.tensor, candidate_attn_mask: torch.tensor,
                candidate_category: torch.tensor, history_encoding: torch.tensor, history_attn_mask: torch.tensor,
                history_category_mask: torch.tensor):
        """
        Forward propagation
        :param candidate_encoding: shape [batch_size, neg_pos_ratio + 1, title_length]
        :param candidate_attn_mask: BoolTensor, shape [batch_size, neg_pos_ratio + 1, title_length]
        :param candidate_category: shape [batch_size, neg_pos_ratio + 1]
        :param history_encoding: shape [batch_size, his_length, title_length]
        :param history_attn_mask: BoolTensor, shape [batch_size, his_length, title_length]
        :param history_category_mask: BoolTensor, shape [batch_size, num_category, his_mask]
        :return:
        """
        # Representation of the candidate news
        batch_size = candidate_encoding.shape[1]
        num_candidates = candidate_encoding.shape[0]
        candidate_encoding = candidate_encoding.view(batch_size * num_candidates, -1)
        candidate_attn_mask = candidate_attn_mask.view(batch_size * num_candidates, -1)
        candidate_news_repr = self.news_encoder(encoding=candidate_encoding, attention_mask=candidate_attn_mask)
        candidate_news_repr = candidate_news_repr.view(batch_size, num_candidates, -1)
        candidate_news_repr = self.dropout(candidate_news_repr)

        # Representation of the users
        category_repr, user_repr = self.user_encoder(encoding=history_encoding, attention_mask=history_attn_mask,
                                                     category_mask=history_category_mask)

        # Calculate interest scores
        user_score = torch.bmm(candidate_news_repr, user_repr)




