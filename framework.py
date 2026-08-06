"""Backward-compatible imports for the negotiation protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation.protocol import Agent, MultiAgentNegotiationFramework, VoteResult

__all__ = ["Agent", "VoteResult", "MultiAgentNegotiationFramework"]
