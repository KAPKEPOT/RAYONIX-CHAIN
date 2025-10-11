# core/dependencies.py - NodeDependencies container

from dataclasses import dataclass
from typing import Optional

@dataclass
class NodeDependencies:
    """Container for node dependencies to enable dependency injection"""
    config_manager: 'ConfigManager'
    rayonix_chain: 'RayonixBlockchain'
    wallet: Optional['RayonixWallet'] = None
    network: Optional['AdvancedP2PNetwork'] = None
    contract_manager: Optional['ContractManager'] = None
    database: Optional['AdvancedDatabase'] = None