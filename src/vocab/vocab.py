from typing import Dict, List

from src.schema.event import Event


class Vocab:
    """
    Simple string-to-index vocabulary with <pad> and <unk>.
    """

    def __init__(self):
        self.stoi: Dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
        }
        self.itos: List[str] = ["<pad>", "<unk>"]

    def add(self, token: str) -> int:
        if token not in self.stoi:
            idx = len(self.itos)
            self.stoi[token] = idx
            self.itos.append(token)
        return self.stoi[token]

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])

    def __len__(self) -> int:
        return len(self.itos)


def build_vocabs(events: List[Event]) -> Dict[str, Vocab]:
    """
    Build vocabularies from a list of events.
    """
    vocabs = {
        "event_type": Vocab(),
        "actor": Vocab(),
        "theme": Vocab(),
        "constraint": Vocab(),
        "action": Vocab(),
    }

    for event in events:
        vocabs["event_type"].add(event.event_type)
        vocabs["action"].add(event.action)

        for actor in event.actors:
            vocabs["actor"].add(actor)

        for theme in event.themes:
            vocabs["theme"].add(theme)

        for constraint in event.constraints:
            vocabs["constraint"].add(constraint)

    return vocabs
