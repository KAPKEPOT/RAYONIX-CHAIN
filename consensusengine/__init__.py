# consensus/__init__.py
"""
Production-ready Proof-of-Stake Consensus Engine
"""

__version__ = "1.0.0"
__author__ = "Consensus Team"

from consensus.core.engine import ProofOfStake
from consensus.models.validators import Validator, ValidatorStatus
from consensus.models.blocks import BlockProposal
from consensus.models.votes import Vote, VoteType
from consensus.abci.interface import ABCIApplication

__all__ = [
    'ProofOfStake',
    'Validator',
    'ValidatorStatus',
    'BlockProposal',
    'Vote',
    'VoteType',
    'ABCIApplication'
]