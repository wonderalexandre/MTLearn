"""CFP valuation facade used in filter specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CFPValuation:
    """Signal reconstructed by one CFP filter.

    ``ALTITUDE`` reconstructs the filtered image altitude, ``ALTITUDE_TOPHAT``
    reconstructs the tree-type-specific altitude top-hat, and
    ``node_attribute`` reconstructs a scalar node attribute.
    """

    kind: str
    attribute: Any = None

    @classmethod
    def node_attribute(cls, attribute: Any) -> "CFPValuation":
        """Use a scalar node attribute as the valuation to be filtered."""
        return cls("node_attribute", attribute)


CFPValuation.ALTITUDE = CFPValuation("altitude")
CFPValuation.ALTITUDE_TOPHAT = CFPValuation("altitude_tophat")
