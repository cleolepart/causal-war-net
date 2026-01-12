from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Event:
    id: str
    date: str
    event_type: str
    actors: List[str]
    themes: List[str]
    constraints: List[str]
    beliefs: Dict[str, str]
    action: str
    escalation_delta: float
    risk_label: str
