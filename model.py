import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.image_encoder = ImageEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.interaction_model = InteractionModel(cfg)
        self.softmax_temperature = nn.Parameter(torch.tensor(0.2))

    def forward(self, object_positions, object_embeddings, text_ids, text_masks):
        '''
        :param object_positions: [B1, S1, 4]
        :param object_embeddings:  [B1, S1, e_ori]
        :param text_ids: [B2, S2]
        :param text_masks: [B2, S2]
        :return: [B2, B1]
        '''
        image_features = self.image_encoder(object_positions, object_embeddings)
        text_features = self.text_encoder(text_ids, text_masks)
        score_matrix = self.interaction_model(image_features, text_features, text_masks)
        score_matrix = torch.softmax(score_matrix / self.softmax_temperature, dim=-1)
        return score_matrix


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.position_embedding_layer = nn.Linear(cfg.data.num_image_position, cfg.model.embedding_size)
        self.layer_norm = nn.LayerNorm(cfg.model.embedding_size)
        self.dropout = nn.Dropout(cfg.model.image_embedding_dropout_rate)
        self.linear_projection = nn.Linear(cfg.model.embedding_size, cfg.model.embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.model.embedding_size, nhead=cfg.model.num_image_encoder_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.num_image_encoder_layer)

    def forward(self, object_positions, object_embeddings):
        '''
        :param object_positions:  [B, S, 4]
        :param object_embeddings:   [B, S, e_ori]
        :return: [B, S, E]
        '''
        position_embeddings = self.position_embedding_layer(object_positions)
        object_embeddings = self.linear_projection(object_embeddings)
        image_embeddings = self.layer_norm(object_embeddings + position_embeddings)
        image_embeddings = self.dropout(image_embeddings)
        return self.encoder(image_embeddings)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.permute(pe, (1, 0, 2))  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_size = cfg.model.embedding_size
        self.word_embeddings = nn.Embedding(cfg.data.max_len, cfg.model.embedding_size)
        self.pos_encoder = PositionalEncoding(cfg.model.embedding_size, cfg.model.text_embedding_dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(d_model=cfg.model.embedding_size, nhead=cfg.model.num_text_encoder_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, cfg.model.num_text_encoder_layer)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, text_ids, text_masks):
        '''
        :param text_ids: [B, S]
        :param text_masks: [B, S]
        :return: [B, S, E]
        '''
        text_features = self.word_embeddings(text_ids)
        text_features = self.pos_encoder(text_features)
        output = self.transformer_encoder(text_features, src_key_padding_mask=text_masks)
        return output


class InteractionModel(nn.Module):
    def __init__(self):
        super(InteractionModel, self).__init__()
        self.softmax_temperature = nn.Parameter(torch.tensor(0.2))

    def forward(self, image_features, text_features, text_masks):
        '''
        :param image_features: [B1, S1, E]
        :param text_features: [B2, S2, E]
        :param text_masks: [B2, S2]
        :return: [B2, B1]
        '''
        unsqueezed_image_features = torch.unsqueeze(image_features, dim=0)  # [1, B1, S1, E]
        unsqueezed_text_features = torch.unsqueeze(text_features, dim=1)  # [B2, 1, S2, E]

        unsqueezed_normalized_image_features = F.normalize(unsqueezed_image_features, p=2, dim=-1)  # [1, B1, S1, E]
        unsqueezed_normalized_text_features = F.normalize(unsqueezed_text_features, p=2, dim=-1)  # [B2, 1, S2, E]

        similarity_logits = torch.matmul(unsqueezed_normalized_text_features, torch.permute(unsqueezed_normalized_image_features, (0, 1, 3, 2)))
        # [B2, B1, S2, S1]

        similarity_scores = torch.softmax(similarity_logits / self.softmax_temperature, dim=-1)  # (B2, B1, S2, S1)
        clamped_similarity_scores = torch.clamp(similarity_scores, min=0)  # (B2, B1, S2, S1)  masked_part = 0,  min = 0

        attn_image_features = torch.matmul(clamped_similarity_scores, unsqueezed_image_features)  # (B2, B1, S2, E)
        normalized_attn_image_features = F.normalize(attn_image_features, p=2, dim=-1)  # (B2, B1, S2, E)
        attn_similarity_scores = torch.sum(unsqueezed_normalized_text_features * normalized_attn_image_features, dim=-1)  # (B2, B1, S2)
        unsqueezed_text_masks = torch.unsqueeze(text_masks, dim=1)  # [B2, 1, S2]
        text_masked_attn_similarity_scores = unsqueezed_text_masks * attn_similarity_scores  # (B2, B1, S2), masked_part = 0

        return torch.mean(text_masked_attn_similarity_scores, dim=-1)  # [B2, B1]

