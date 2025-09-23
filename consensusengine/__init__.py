# consensus/__init__.py
"""
Production-ready Proof-of-Stake Consensus Engine
"""

__version__ = "1.0.0"
__author__ = "Consensus Team"

from consensusengine.core.engine import ProofOfStake
from consensusengine.models.validators import Validator, ValidatorStatus
from consensusengine.models.blocks import BlockProposal
from consensusengine.models.votes import Vote, VoteType
from consensusengine.abci.interface import ABCIApplication

__all__ = [
    'ProofOfStake',
    'Validator',
    'ValidatorStatus',
    'BlockProposal',
    'Vote',
    'VoteType',
    'ABCIApplication'
]