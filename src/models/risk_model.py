from typing import Dict

import torch
import torch.nn as nn

from src.models.encoder import EventEncoder


class WWIRiskModel(nn.Module):
    """
    Predicts escalation risk from a sequence of encoded events.
    """

    def __init__(
        self,
        encoder: EventEncoder,
        d_model: int,
        n_classes: int = 3,
    ):
        super().__init__()

        self.encoder = encoder
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch keys:
          - event_type, action, actors, themes, constraints, mask

        returns:
          - logits: [B, n_classes]
        """
        # Encode events
        x = self.encoder(batch)  # [B, T, D]

        # Temporal modeling
        out, _ = self.rnn(x)  # [B, T, D]

        # Select last valid timestep using mask
        lengths = batch["mask"].sum(dim=1) - 1  # [B]
        last = out[torch.arange(out.size(0)), lengths]  # [B, D]

        return self.head(last)  # [B, n_classes]
