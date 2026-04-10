"""GPU-first full M3Net architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import M3NetFullConfig

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass
class M3NetFullOutput:
    return_pred: "torch.Tensor"
    risk_pred: "torch.Tensor"
    confidence: "torch.Tensor"
    confidence_logits: "torch.Tensor"
    top_pick_prob: "torch.Tensor"
    top_pick_logits: "torch.Tensor"
    fused_embedding: "torch.Tensor"
    graph_projection: "torch.Tensor"
    graph_positive_mask: "torch.Tensor"


if nn is not None:

    class TemporalEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            hidden = self.input_proj(x)
            encoded = self.encoder(hidden)
            return self.norm(encoded[:, -1, :])


    class CrossSectionFusion(nn.Module):
        def __init__(self, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.norm(self.encoder(x))


    class GraphRelationLayer(nn.Module):
        def __init__(self, hidden_dim: int, neighbor_k: int, residual_weight: float, temperature: float) -> None:
            super().__init__()
            self.neighbor_k = max(1, int(neighbor_k))
            self.residual_weight = float(residual_weight)
            self.temperature = float(max(temperature, 1e-6))
            self.message_proj = nn.Linear(hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, fused: "torch.Tensor", relation_source: "torch.Tensor") -> "torch.Tensor":
            num_symbols = fused.size(0)
            if num_symbols <= 1:
                return fused

            source = F.normalize(relation_source, dim=-1)
            similarity = torch.matmul(source, source.transpose(0, 1)) / self.temperature
            similarity = similarity.masked_fill(torch.eye(num_symbols, device=similarity.device, dtype=torch.bool), float("-inf"))

            neighbor_k = min(self.neighbor_k, max(num_symbols - 1, 1))
            top_values, top_indices = torch.topk(similarity, k=neighbor_k, dim=-1)
            attention = torch.softmax(top_values, dim=-1)
            neighbor_states = fused[top_indices]
            messages = (attention.unsqueeze(-1) * neighbor_states).sum(dim=1)
            messages = self.message_proj(messages)
            return self.norm(fused + self.residual_weight * messages)


    class M3NetFullModel(nn.Module):
        """High-capacity M3Net for GPU training."""

        def __init__(self, config: Optional[M3NetFullConfig] = None) -> None:
            super().__init__()
            self.config = config or M3NetFullConfig()
            self.daily_encoder = TemporalEncoder(
                input_dim=self.config.daily_input_dim,
                hidden_dim=self.config.hidden_dim,
                layers=self.config.daily_layers,
                heads=self.config.attention_heads,
                dropout=self.config.dropout,
            )
            self.intraday_encoder = TemporalEncoder(
                input_dim=self.config.intraday_input_dim,
                hidden_dim=self.config.hidden_dim,
                layers=self.config.intraday_layers,
                heads=self.config.attention_heads,
                dropout=self.config.dropout,
            )
            self.factor_proj = nn.Sequential(
                nn.Linear(self.config.factor_input_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.config.hidden_dim),
                nn.Dropout(self.config.dropout),
            )
            self.memory_proj = nn.Sequential(
                nn.Linear(self.config.memory_input_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.config.hidden_dim),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.config.hidden_dim * 4, self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, 4),
            )
            self.cross_section_fusion = CrossSectionFusion(
                hidden_dim=self.config.hidden_dim,
                layers=self.config.fusion_layers,
                heads=self.config.attention_heads,
                dropout=self.config.dropout,
            )
            self.graph_relation = GraphRelationLayer(
                hidden_dim=self.config.hidden_dim,
                neighbor_k=self.config.graph_neighbor_k,
                residual_weight=self.config.graph_residual_weight,
                temperature=self.config.graph_temperature,
            )
            self.return_head = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.GELU(), nn.Linear(self.config.hidden_dim, 1))
            self.risk_head = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.GELU(), nn.Linear(self.config.hidden_dim, 1))
            self.confidence_head = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.GELU(), nn.Linear(self.config.hidden_dim, 1))
            self.top_pick_head = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.GELU(), nn.Linear(self.config.hidden_dim, 1))
            self.graph_projection_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            )

        def _build_graph_positive_mask(self, relation_source: "torch.Tensor") -> "torch.Tensor":
            num_symbols = relation_source.size(0)
            if num_symbols <= 1:
                return torch.zeros((num_symbols, num_symbols), device=relation_source.device, dtype=torch.bool)

            source = F.normalize(relation_source, dim=-1)
            similarity = torch.matmul(source, source.transpose(0, 1)) / max(self.config.graph_contrastive_temperature, 1e-6)
            similarity = similarity.masked_fill(torch.eye(num_symbols, device=similarity.device, dtype=torch.bool), float("-inf"))
            neighbor_k = min(max(1, self.config.graph_contrastive_neighbors), max(num_symbols - 1, 1))
            _, top_indices = torch.topk(similarity, k=neighbor_k, dim=-1)
            positive_mask = torch.zeros((num_symbols, num_symbols), device=relation_source.device, dtype=torch.bool)
            positive_mask.scatter_(1, top_indices, True)
            return positive_mask

        def forward(
            self,
            daily_sequence: "torch.Tensor",
            intraday_sequence: "torch.Tensor",
            factor_features: "torch.Tensor",
            memory_features: "torch.Tensor",
        ) -> M3NetFullOutput:
            daily_latent = self.daily_encoder(daily_sequence)
            intraday_latent = self.intraday_encoder(intraday_sequence)
            factor_latent = self.factor_proj(factor_features)
            memory_latent = self.memory_proj(memory_features).unsqueeze(0).expand(daily_latent.size(0), -1)
            graph_positive_mask = self._build_graph_positive_mask(factor_latent)

            gate_input = torch.cat([daily_latent, intraday_latent, factor_latent, memory_latent], dim=-1)
            gate_logits = self.fusion_gate(gate_input)
            gate_weights = torch.softmax(gate_logits, dim=-1)
            fused = (
                gate_weights[:, 0:1] * daily_latent
                + gate_weights[:, 1:2] * intraday_latent
                + gate_weights[:, 2:3] * factor_latent
                + gate_weights[:, 3:4] * memory_latent
            )
            fused = self.graph_relation(fused, factor_latent)
            fused = self.cross_section_fusion(fused.unsqueeze(0)).squeeze(0)

            confidence_logits = self.confidence_head(fused).squeeze(-1)
            top_pick_logits = self.top_pick_head(fused).squeeze(-1)
            graph_projection = F.normalize(self.graph_projection_head(fused), dim=-1)
            return M3NetFullOutput(
                return_pred=self.return_head(fused).squeeze(-1),
                risk_pred=F.softplus(self.risk_head(fused).squeeze(-1)),
                confidence=torch.sigmoid(confidence_logits),
                confidence_logits=confidence_logits,
                top_pick_prob=torch.sigmoid(top_pick_logits),
                top_pick_logits=top_pick_logits,
                fused_embedding=fused,
                graph_projection=graph_projection,
                graph_positive_mask=graph_positive_mask,
            )

        def loss(
            self,
            output: M3NetFullOutput,
            future_return: "torch.Tensor",
            future_risk: "torch.Tensor",
        ) -> "torch.Tensor":
            centered_return = future_return - future_return.mean()
            scaled_return = centered_return / future_return.std(unbiased=False).clamp_min(1e-6)
            return_rank = torch.argsort(torch.argsort(future_return)).float()
            return_rank = return_rank / max(float(future_return.numel() - 1), 1.0)
            sample_weight = 1.0 + self.config.weighted_return_alpha * return_rank
            return_loss = (F.smooth_l1_loss(output.return_pred, scaled_return, reduction="none") * sample_weight).mean()
            risk_loss = F.mse_loss(output.risk_pred, future_risk)

            if future_return.numel() > 1:
                pairwise_gap = future_return.unsqueeze(1) - future_return.unsqueeze(0)
                pairwise_target = torch.sign(pairwise_gap)
                pairwise_score = output.return_pred.unsqueeze(1) - output.return_pred.unsqueeze(0)
                pairwise_weight = pairwise_gap.abs().pow(self.config.rank_gap_power)
                pairwise_mask = pairwise_target.ne(0)
                if pairwise_mask.any():
                    rank_loss = (
                        F.softplus(-pairwise_target[pairwise_mask] * pairwise_score[pairwise_mask]) * pairwise_weight[pairwise_mask]
                    ).mean()
                else:
                    rank_loss = future_return.new_tensor(0.0)

                order = torch.argsort(future_return, descending=True)
                ordered_scores = output.return_pred[order]
                ordered_count = ordered_scores.size(0)
                listmle_terms = torch.logcumsumexp(torch.flip(ordered_scores, dims=[0]), dim=0)
                listmle_terms = torch.flip(listmle_terms, dims=[0]) - ordered_scores
                discounts = 1.0 / torch.log2(torch.arange(ordered_count, device=future_return.device, dtype=future_return.dtype) + 2.0)
                focus = max(1, min(self.config.listwise_topk_focus, ordered_count))
                if focus < ordered_count:
                    discounts[focus:] = discounts[focus:] * self.config.listwise_tail_weight
                listwise_loss = (listmle_terms * discounts).sum() / discounts.sum().clamp_min(1e-6)
            else:
                rank_loss = future_return.new_tensor(0.0)
                listwise_loss = future_return.new_tensor(0.0)

            confidence_target = (future_return > 0).float()
            confidence_loss = F.binary_cross_entropy_with_logits(output.confidence_logits, confidence_target)
            top_pick_threshold = torch.quantile(future_return.detach(), self.config.top_pick_quantile)
            top_pick_target = (future_return >= top_pick_threshold).float()
            positive_count = top_pick_target.sum().clamp_min(1.0)
            negative_count = (1.0 - top_pick_target).sum().clamp_min(1.0)
            pos_weight = negative_count / positive_count
            top_pick_loss = F.binary_cross_entropy_with_logits(
                output.top_pick_logits,
                top_pick_target,
                pos_weight=pos_weight,
            )

            if output.graph_projection.size(0) > 1 and output.graph_positive_mask.any():
                similarity = torch.matmul(output.graph_projection, output.graph_projection.transpose(0, 1))
                similarity = similarity / max(self.config.graph_contrastive_temperature, 1e-6)
                self_mask = torch.eye(similarity.size(0), device=similarity.device, dtype=torch.bool)
                similarity = similarity.masked_fill(self_mask, float("-inf"))
                positive_mask = output.graph_positive_mask & (~self_mask)
                valid_mask = positive_mask.sum(dim=1) > 0
                log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
                masked_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
                graph_contrastive_loss = -(masked_log_prob.sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0))
                graph_contrastive_loss = graph_contrastive_loss[valid_mask].mean() if valid_mask.any() else future_return.new_tensor(0.0)
            else:
                graph_contrastive_loss = future_return.new_tensor(0.0)

            return (
                self.config.return_loss_weight * return_loss
                + self.config.rank_loss_weight * rank_loss
                + self.config.listwise_loss_weight * listwise_loss
                + self.config.risk_loss_weight * risk_loss
                + self.config.confidence_loss_weight * confidence_loss
                + self.config.top_pick_loss_weight * top_pick_loss
                + self.config.graph_contrastive_loss_weight * graph_contrastive_loss
            )

else:

    class M3NetFullModel:  # pragma: no cover - exercised only when torch is absent
        def __init__(self, config: Optional[M3NetFullConfig] = None) -> None:
            raise ImportError("PyTorch is required for the full M3Net model.")
