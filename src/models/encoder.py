from typing import Dict

import torch
import torch.nn as nn


class EventEncoder(nn.Module):
    """
    Encodes symbolic event fields into dense event embeddings.

    Each event is represented by:
      - event_type embedding
      - action embedding
      - averaged actor embeddings
      - averaged theme embeddings
      - averaged constraint embeddings
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        d_model: int = 128,
    ):
        super().__init__()

        self.d_model = d_model

        self.event_type_emb = nn.Embedding(
            vocab_sizes["event_type"], d_model
        )
        self.action_emb = nn.Embedding(
            vocab_sizes["action"], d_model
        )
        self.actor_emb = nn.Embedding(
            vocab_sizes["actor"], d_model
        )
        self.theme_emb = nn.Embedding(
            vocab_sizes["theme"], d_model
        )
        self.constraint_emb = nn.Embedding(
            vocab_sizes["constraint"], d_model
        )

        # Project concatenated embeddings back to d_model
        self.proj = nn.Linear(d_model * 5, d_model)

    def _avg_set(
        self,
        emb: nn.Embedding,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Average embeddings over a padded set dimension.

        x: Tensor of shape [B, T, K]
        returns: Tensor of shape [B, T, d_model]
        """
        # Embed: [B, T, K, d]
        v = emb(x)

        # Mask padding (pad_id = 0)
        mask = (x != 0).unsqueeze(-1)  # [B, T, K, 1]
        v = v * mask

        denom = mask.sum(dim=2).clamp(min=1)  # avoid divide-by-zero
        return v.sum(dim=2) / denom

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of event sequences.

        batch keys:
          - event_type: [B, T]
          - action: [B, T]
          - actors: [B, T, K]
          - themes: [B, T, K]
          - constraints: [B, T, K]

        returns:
          - event_embeddings: [B, T, d_model]
        """
        et = self.event_type_emb(batch["event_type"])     # [B, T, d]
        ac = self.action_emb(batch["action"])             # [B, T, d]
        aa = self._avg_set(self.actor_emb, batch["actors"])
        th = self._avg_set(self.theme_emb, batch["themes"])
        co = self._avg_set(self.constraint_emb, batch["constraints"])

        # Concatenate along feature dimension
        x = torch.cat([et, ac, aa, th, co], dim=-1)  # [B, T, 5d]

        return self.proj(x)  # [B, T, d_model]
