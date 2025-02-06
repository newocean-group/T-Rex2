from src.common_modules import (
    AggregatingFeatures,
    AggregatingFeaturesForVisualEncoder,
    DeformableTransformerLayerForVisualEncoder,
    DeformableTransformerForVisualEncoder,
)
import torch.nn as nn
from ops.modules import MSDeformAttn
from torch.nn.init import normal_
import torch


class VisualPromptEncoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        nhead=8,
        num_layers=3,
        n_points=4,
        dropout=0.0,
        n_levels=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        # using for capturing feats from ref boxes
        self.content_embeddings = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        normal_(self.content_embeddings.weight.data)

        self.cls_embeddings = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        normal_(self.cls_embeddings.weight.data)

        encoder_layer = DeformableTransformerLayerForVisualEncoder(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            n_levels=n_levels,
            n_heads=nhead,
            n_points=n_points,
        )

        self.visual_encoder = DeformableTransformerForVisualEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            nhead=nhead,
            d_model=d_model,
        )
        self.global_coords = torch.tensor([[0.5, 0.5, 1.0, 1.0]], requires_grad=False)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def create_aggregating_features_for_visual_encoder(self):
        aggregating_features_layer = AggregatingFeatures(
            nhead=self.n_heads,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
        )
        self.aggregating_features_layers = AggregatingFeaturesForVisualEncoder(
            aggregating_features_layer, num_layers=self.num_layers
        )

    # def forward(
    #     self,
    #     memory,
    #     mask_flatten,
    #     spatial_shapes,
    #     level_start_index,
    #     valid_ratios,
    #     reference_points,
    #     reference_points_mask,
    # ):
    #     """
    #     memory shape: bs x L x c
    #     reference_points shape: bs x K x 4
    #     reference_points_mask shape: bs x K
    #     """
    #     bs, _, _ = memory.shape
    #     _, K, _ = reference_points.shape
    #     content_embeddings = self.content_embeddings.weight
    #     tgt = content_embeddings[None].repeat(bs, K, 1)

    #     # visual prompt embeddings
    #     visual_embeddings = self.visual_encoder(
    #         tgt=tgt,
    #         reference_points=reference_points,
    #         src=memory,
    #         src_spatial_shapes=spatial_shapes,
    #         src_level_start_index=level_start_index,
    #         src_valid_ratios=valid_ratios,
    #         query_pos=None,
    #         src_padding_mask=mask_flatten,
    #         reference_points_mask=reference_points_mask,
    #     )  # bs x K x d_model

    #     # aggregating features
    #     cls_embeddings = self.cls_embeddings.weight
    #     tgt = torch.cat(
    #         [
    #             visual_embeddings,
    #             cls_embeddings[None].repeat(bs, 1, 1),
    #         ],
    #         dim=1,
    #     )  # bs x K + 1 x d_model

    #     visual_embeddings = self.aggregating_features_layers(tgt)

    #     return visual_embeddings  # bs x K + 1 x d_model

    def forward(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        reference_points,
        reference_points_mask,
    ):
        """
        memory shape: bs x L x c
        reference_points shape: bs x K x 4
        reference_points_mask shape: bs x K
        """
        bs, _, _ = memory.shape
        _, K, _ = reference_points.shape
        content_embeddings = self.content_embeddings.weight
        tgt = content_embeddings[None].repeat(bs, K, 1)

        # visual prompt embeddings
        visual_embeddings = self.visual_encoder(
            tgt=tgt,
            reference_points=reference_points,
            src=memory,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=None,
            src_padding_mask=mask_flatten,
            reference_points_mask=reference_points_mask,
        )  # bs x K x d_model

        # aggregating features
        avg_visual_embeddings = visual_embeddings.mean(dim=1, keepdim=True)

        visual_embeddings = torch.cat([visual_embeddings, avg_visual_embeddings], dim=1)

        return visual_embeddings  # bs x K + 1 x d_model

    # def forward(
    #     self,
    #     memory,
    #     mask_flatten,
    #     spatial_shapes,
    #     level_start_index,
    #     valid_ratios,
    #     reference_points,
    #     reference_points_mask,
    # ):
    #     """
    #     memory shape: bs x L x c
    #     reference_points shape: bs x K x 4
    #     reference_points_mask shape: bs x K
    #     """
    #     bs, _, _ = memory.shape
    #     _, K, _ = reference_points.shape

    #     content_embeddings = self.content_embeddings.weight
    #     cls_embeddings = self.cls_embeddings.weight
    #     tgt = torch.cat(
    #         [
    #             content_embeddings[None].repeat(bs, K, 1),
    #             cls_embeddings[None].repeat(bs, 1, 1),
    #         ],
    #         dim=1,
    #     )  # bs x K + 1 x d_model

    #     new_reference_points = torch.cat(
    #         [
    #             reference_points,
    #             self.global_coords.repeat(bs, 1)[:, None].to(memory.device),
    #         ],
    #         dim=1,
    #     )  # bs x K + 1 x 4

    #     new_reference_points_mask = torch.cat(
    #         [
    #             reference_points_mask,
    #             torch.zeros_like(reference_points_mask)[:, :1].to(memory.device),
    #         ],
    #         dim=1,
    #     )  # bs x K + 1

    #     # visual prompt embeddings
    #     visual_embeddings = self.visual_encoder(
    #         tgt=tgt,
    #         reference_points=new_reference_points,
    #         src=memory,
    #         src_spatial_shapes=spatial_shapes,
    #         src_level_start_index=level_start_index,
    #         src_valid_ratios=valid_ratios,
    #         query_pos=None,
    #         src_padding_mask=mask_flatten,
    #         reference_points_mask=new_reference_points_mask,
    #     )  # bs x K + 1 x d_model
    #     return visual_embeddings  # bs x K + 1 x d_model
