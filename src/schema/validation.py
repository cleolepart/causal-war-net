from typing import Iterable

from src.schema.event import Event


ALLOWED_RISK_LABELS = {"low", "medium", "high"}


def validate_event(event: Event) -> None:
    """
    Raise ValueError if the event violates schema or semantic constraints.
    """

    if event.risk_label not in ALLOWED_RISK_LABELS:
        raise ValueError(
            f"Invalid risk_label '{event.risk_label}' in event {event.id}"
        )

    if not event.actors:
        raise ValueError(
            f"Event {event.id} must have at least one actor"
        )

    if not event.themes:
        raise ValueError(
            f"Event {event.id} must have at least one theme"
        )

    if not isinstance(event.escalation_delta, (int, float)):
        raise ValueError(
            f"Event {event.id} escalation_delta must be numeric"
        )

    if not (-10.0 < event.escalation_delta < 10.0):
        raise ValueError(
            f"Event {event.id} escalation_delta out of reasonable bounds"
        )


def validate_events(events: Iterable[Event]) -> None:
    """
    Validate a sequence of events.
    """
    for event in events:
        validate_event(event)
