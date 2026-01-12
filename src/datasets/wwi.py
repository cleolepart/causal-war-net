import json
from typing import List

from src.schema.event import Event
from src.schema.validation import validate_event



def load_ww1_events(path: str) -> List[Event]:
    """
    Load WWI events from a JSONL file.
    Each line must be a single JSON object matching the Event schema.
    """
    events: List[Event] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: {e}"
                ) from e

            try:
                event = Event(**obj)
            except TypeError as e:
                raise ValueError(
                    f"Schema mismatch on line {line_num}: {e}"
                ) from e
            
            validate_event(event)
            events.append(event)

    return events

from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset

from src.schema.event import Event
from src.vocab.vocab import Vocab


RISK_TO_ID = {"low": 0, "medium": 1, "high": 2}


class WWIDataset(Dataset):
    """
    Dataset producing (event sequence prefix → risk label) samples.
    """

    def __init__(
        self,
        events: List[Event],
        vocabs: Dict[str, Vocab],
        max_len: int = 32,
        max_set_size: int = 8,
    ):
        self.events = events
        self.vocabs = vocabs
        self.max_len = max_len
        self.max_set_size = max_set_size

    def __len__(self) -> int:
        return len(self.events)

    def _encode_event(self, event: Event) -> Dict[str, Any]:
        return {
            "event_type": self.vocabs["event_type"].encode(event.event_type),
            "action": self.vocabs["action"].encode(event.action),
            "actors": [
                self.vocabs["actor"].encode(a)
                for a in event.actors[: self.max_set_size]
            ],
            "themes": [
                self.vocabs["theme"].encode(t)
                for t in event.themes[: self.max_set_size]
            ],
            "constraints": [
                self.vocabs["constraint"].encode(c)
                for c in event.constraints[: self.max_set_size]
            ],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start = max(0, idx - self.max_len + 1)
        prefix = self.events[start : idx + 1]

        encoded = [self._encode_event(e) for e in prefix]

        label = RISK_TO_ID[self.events[idx].risk_label]

        return {
            "sequence": encoded,
            "y": label,
        }


def collate_wwi(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate variable-length WWI event sequences into padded tensors.
    """

    batch_size = len(batch)
    max_T = max(len(item["sequence"]) for item in batch)
    K = max(
        max(len(e["actors"]), len(e["themes"]), len(e["constraints"]))
        for item in batch
        for e in item["sequence"]
    )

    pad_id = 0

    event_type = torch.full((batch_size, max_T), pad_id, dtype=torch.long)
    action = torch.full((batch_size, max_T), pad_id, dtype=torch.long)
    actors = torch.full((batch_size, max_T, K), pad_id, dtype=torch.long)
    themes = torch.full((batch_size, max_T, K), pad_id, dtype=torch.long)
    constraints = torch.full((batch_size, max_T, K), pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_T), dtype=torch.bool)
    y = torch.tensor([item["y"] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        seq = item["sequence"]
        mask[i, : len(seq)] = True

        for t, e in enumerate(seq):
            event_type[i, t] = e["event_type"]
            action[i, t] = e["action"]

            actors[i, t, : len(e["actors"])] = torch.tensor(
                e["actors"], dtype=torch.long
            )
            themes[i, t, : len(e["themes"])] = torch.tensor(
                e["themes"], dtype=torch.long
            )
            constraints[i, t, : len(e["constraints"])] = torch.tensor(
                e["constraints"], dtype=torch.long
            )

    return {
        "event_type": event_type,
        "action": action,
        "actors": actors,
        "themes": themes,
        "constraints": constraints,
        "mask": mask,
        "y": y,
    }
