"""Registry for serializable CFP extension components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class SpecRegistry:
    """Map config ``kind`` strings to component factories."""

    def __init__(self):
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, kind: str, factory: Callable[..., Any]) -> None:
        """Register a component factory."""
        kind = str(kind)
        if not kind:
            raise ValueError("kind must be non-empty.")
        if not callable(factory):
            raise TypeError("factory must be callable.")

        if kind in self._factories:
            raise ValueError(f"CFP component kind is already registered: {kind!r}")
        self._factories[kind] = factory

    def resolve(self, kind: str) -> Callable[..., Any]:
        """Return the factory registered for ``kind``."""
        try:
            return self._factories[str(kind)]
        except KeyError as exc:
            raise KeyError(f"unknown CFP component kind: {kind!r}") from exc

    def create(self, config: Mapping[str, Any], **context):
        """Instantiate a registered component from a config mapping."""
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        kind = config.get("kind")
        factory = self.resolve(kind)
        kwargs = {key: value for key, value in config.items() if key != "kind"}
        return factory(**context, **kwargs)

    def registered_kinds(self) -> tuple[str, ...]:
        """Return registered config kind names."""
        return tuple(sorted(self._factories))
