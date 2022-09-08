import torch
import torch.nn as nn

from src.model.news_encoder import NewsEncoder
from src.model.user_encoder import UserEncoder


class HieRec(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, user_encoder: UserEncoder, dropout: float, score_weight: float):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.dropout = nn.Dropout(dropout)
        self.user_score_weight = score_weight

    def forward(self, candidate_encoding: torch.tensor, candidate_attn_mask: torch.tensor,
                candidate_category_mask: torch.tensor, history_encoding: torch.tensor, history_attn_mask: torch.tensor,
                history_category_mask: torch.tensor):
        """
        Forward propagation
        :param candidate_encoding: shape [batch_size, neg_pos_ratio + 1, title_length]
        :param candidate_attn_mask: BoolTensor, shape [batch_size, neg_pos_ratio + 1, title_length]
        :param candidate_category_mask: BoolTensor, shape [batch_size, neg_pos_ratio + 1, num_category]
        :param history_encoding: shape [batch_size, his_length, title_length]
        :param history_attn_mask: BoolTensor, shape [batch_size, his_length, title_length]
        :param history_category_mask: BoolTensor, shape [batch_size, num_category, his_length]
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
        user_score = torch.bmm(candidate_news_repr, user_repr.unsqueeze(dim=2)).squeeze(dim=2)
        category_score = torch.bmm(candidate_category_mask, category_repr)
        category_score = torch.sum(candidate_news_repr * category_score, dim=2)
        num_history_click = torch.sum(history_category_mask, dim=(1, 2)).unsqueeze(dim=1)
        history_category_count = torch.sum(history_category_mask, dim=2)
        history_category_ratio = history_category_count / num_history_click
        candidate_category_ratio = torch.bmm(candidate_category_mask, history_category_ratio.unsqueeze(dim=2))\
            .squeeze(dim=2)
        category_score = category_score * candidate_category_ratio

        logits = self.user_score_weight * user_score + (1 - self.user_score_weight) * category_score

        return logits
