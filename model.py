import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.image_encoder = ImageEncoder(cfg)
        # self.image_encoder = IdentityImageEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.interaction_model = InteractionModel(cfg)


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
        return score_matrix


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.position_embedding_layer = nn.Linear(cfg.data.num_image_position, cfg.model.embedding_size)
        nn.init.zeros_(self.position_embedding_layer.weight)
        self.layer_norm = nn.LayerNorm(cfg.model.embedding_size)
        self.dropout = nn.Dropout(cfg.model.image_embedding_dropout_rate)
        self.linear_projection = nn.Linear(cfg.model.embedding_size, cfg.model.embedding_size)
        nn.init.eye_(self.linear_projection.weight)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.model.embedding_size, nhead=cfg.model.num_image_encoder_head, batch_first=True, dropout=cfg.model.image_encoder_dropout_rate)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.num_image_encoder_layer)

    def forward(self, object_positions, object_embeddings):
        '''
        :param object_positions:  [B, S, 4]
        :param object_embeddings:   [B, S, e_ori]
        :return: [B, S, E]
        '''
        position_embeddings = self.position_embedding_layer(object_positions)
        object_embeddings = self.linear_projection(object_embeddings)
        object_embeddings = F.normalize(object_embeddings, dim=-1)
        image_embeddings = self.layer_norm(object_embeddings + 0.1 * position_embeddings)
        image_embeddings = self.dropout(image_embeddings)
        return F.normalize(self.encoder(image_embeddings), dim=-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout_rate=0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_size = cfg.model.embedding_size
        self.word_embeddings = nn.Embedding(cfg.data.vocab_size, cfg.model.embedding_size)
        self.pos_encoder = PositionalEncoding(cfg.model.embedding_size, cfg.model.text_embedding_dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(d_model=cfg.model.embedding_size, nhead=cfg.model.num_text_encoder_head, batch_first=True, dropout=cfg.model.text_encoder_dropout_rate)
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
        return F.normalize(output, dim=-1)


class InteractionModel(nn.Module):
    def __init__(self, cfg):
        super(InteractionModel, self).__init__()
        # self.softmax_temperature = nn.Parameter(torch.tensor(1.0))
        self.softmax_temperature = 0.11

    def forward(self, image_features, text_features, text_masks):
        '''
        :param image_features: [B1, S1, E]
        :param text_features: [B2, S2, E]
        :param text_masks: [B2, S2]
        :return: [B2, B1]
        '''
        unsqueezed_image_features = torch.unsqueeze(image_features, dim=0)  # [1, B1, S1, E]
        unsqueezed_text_features = torch.unsqueeze(text_features, dim=1)  # [B2, 1, S2, E]

        similarity_logits = torch.matmul(unsqueezed_text_features, torch.permute(unsqueezed_image_features, (0, 1, 3, 2)))
        # [B2, B1, S2, S1]

        similarity_scores = F.normalize(similarity_logits.clamp(min=0), dim=-1)
        similarity_scores = torch.softmax(similarity_logits / self.softmax_temperature, dim=-1)  # (B2, B1, S2, S1)

        attn_image_features = torch.matmul(similarity_scores, unsqueezed_image_features)  # (B2, B1, S2, E)
        normalized_attn_image_features = F.normalize(attn_image_features, p=2, dim=-1)  # (B2, B1, S2, E)
        attn_similarity_scores = torch.sum(unsqueezed_text_features * normalized_attn_image_features, dim=-1)  # (B2, B1, S2)

        unsqueezed_text_masks = torch.unsqueeze(text_masks, dim=1)  # [B2, 1, S2]
        text_masked_attn_similarity_scores = unsqueezed_text_masks * attn_similarity_scores  # (B2, B1, S2), masked_part = 0

        mask_num = torch.sum(text_masks, dim=-1, keepdim=True)  # [B2, 1]
        return torch.sum(text_masked_attn_similarity_scores, dim=-1) / mask_num  # [B2, B1]


class IdentityImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(IdentityImageEncoder, self).__init__()
        self.linear_projection = nn.Linear(cfg.model.embedding_size, cfg.model.embedding_size)
        nn.init.eye_(self.linear_projection.weight)

    def forward(self, object_positions, object_embeddings):
        '''
        :param object_positions:  [B, S, 4]
        :param object_embeddings:   [B, S, e_ori]
        :return: [B, S, E]
        '''
        return F.normalize(self.linear_projection(object_embeddings), dim=-1)