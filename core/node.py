# core/node.py - Main RayonixNode class

import asyncio
import threading
from typing import Optional, Set, Any  # Add Any import
from pathlib import Path
import logging

from core.dependencies import NodeDependencies
from core.state_manager import NodeStateManager
from network.network_manager import NetworkManager
from api.server import RayonixAPIServer
from tasks.staking_task import StakingTask
from tasks.mempool_task import MempoolTask
from tasks.peer_monitor import PeerMonitor
from network.sync_manager import SyncManager
from config.patch_config import get_safe_genesis_config

logger = logging.getLogger("rayonix_node.core")

class RayonixNode:
    """Complete RAYONIX blockchain node implementation"""
    
    def __init__(self, dependencies: Optional[NodeDependencies] = None):
        # Use provided dependencies or create empty ones
        if dependencies:
            self.deps = dependencies
            self.config_manager = dependencies.config_manager
            self.rayonix_chain = dependencies.rayonix_coin
            self.wallet = dependencies.wallet
            self.network = dependencies.network
            self.contract_manager = dependencies.contract_manager
            self.database = dependencies.database
        else:
            self.deps = NodeDependencies(
                config_manager=None,
                rayonix_chain=None,
                wallet=None,
                network=None,
                contract_manager=None,
                database=None
            )
            self.config_manager = None
            self.rayonix_chain = None
            self.wallet = None
            self.network = None
            self.contract_manager = None
            self.database = None
        
        self.running = False
        self.shutdown_event = threading.Event()
        self.api_server = None
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Initialize managers
        self.state_manager = NodeStateManager()
        self.network_manager = NetworkManager(self)
        self.sync_manager = SyncManager(self)
        
        # Initialize tasks
        self.staking_task = StakingTask(self)
        self.mempool_task = MempoolTask(self)
        self.peer_monitor = PeerMonitor(self)
    
    async def initialize_components(self, config_path: Optional[str] = None, 
                                  encryption_key: Optional[str] = None) -> bool:
        """Initialize all node components with dependency injection support"""
        try:
            logger.info("Initializing RAYONIX Node components...")
            
            # Initialize config if not provided via dependencies
            if not self.config_manager:
                from config.config_manager import init_config
                self.config_manager = init_config(config_path, encryption_key, auto_reload=True)
            
            # Initialize rayonix_chain if not provided via dependencies
            if not self.rayonix_chain:
                from blockchain.core.rayonix_chain import RayonixBlockchain
                
                network_type = self.config_manager.get('network.network_type', 'testnet')
                data_dir = Path(self.config_manager.get('database.db_path', './rayonix_data'))
                data_dir.mkdir(exist_ok=True, parents=True)
                # Create blockchain configuration from main config
                blockchain_config = {
                    'network_type': network_type,
                    'data_dir': str(data_dir),
                    'port': self.config_manager.get('network.port', 30303),
                    'max_connections': self.config_manager.get('network.max_connections', 50),
                    'block_time_target': self.config_manager.get('consensus.block_time_target', 30),
                    'max_block_size': self.config_manager.get('consensus.max_block_size', 4000000),
                    'min_transaction_fee': self.config_manager.get('gas.min_transaction_fee', 1),
                    'stake_minimum': self.config_manager.get('consensus.stake_minimum', 1000),
                    'developer_fee_percent': self.config_manager.get('consensus.developer_fee_percent', 0.05),
                    'enable_auto_staking': self.config_manager.get('consensus.enable_auto_staking', True),
                    'enable_transaction_relay': self.config_manager.get('network.enable_transaction_relay', True),
                    'enable_state_pruning': self.config_manager.get('database.enable_state_pruning', True),
                    'max_reorganization_depth': self.config_manager.get('consensus.max_reorganization_depth', 100),
                    'checkpoint_interval': self.config_manager.get('database.checkpoint_interval', 1000)
                }
                
                # Get gas price config from main config
                gas_price_config = {
                    'base_gas_price': self.config_manager.get('gas.base_gas_price', 1000000000),
                    'min_gas_price': self.config_manager.get('gas.min_gas_price', 500000000),
                    'max_gas_price': self.config_manager.get('gas.max_gas_price', 10000000000),
                    'adjustment_factor': self.config_manager.get('gas.adjustment_factor', 1.125),
                    'target_utilization': self.config_manager.get('gas.target_utilization', 0.5)
                }
                
                self.rayonix_chain = RayonixBlockchain(
                    network_type=network_type,
                    data_dir=str(data_dir),
                    config=BlockchainConfig
                )
            
            # Initialize wallet if not provided via dependencies
            if not self.wallet:
                await self._initialize_wallet_with_blockchain()
            
            # Initialize network if enabled and not provided via dependencies
            if self.config_manager.get('network.enabled', True) and not self.network:
                await self.network_manager.initialize_network()
                self.network = self.network_manager.network
            
            # Initialize API server if enabled
            if self.config_manager.get('api.enabled', True):
                api_host = self.config_manager.get('api.host', '127.0.0.1')
                api_port = self.config_manager.get('api.port', 8545)
                self.api_server = RayonixAPIServer(self, api_host, api_port)
            
            logger.info("RAYONIX Node components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize node components: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _initialize_wallet_with_blockchain(self):
        """Initialize wallet with proper blockchain reference integration"""
        from rayonix_wallet.core.wallet import RayonixWallet, create_new_wallet
        from rayonix_wallet.core.wallet_config import WalletType, AddressType
        
        wallet_file = Path(self.config_manager.get('database.db_path', './rayonix_data')) / 'wallet.dat'
        if wallet_file.exists():
            try:
                # Load existing wallet
                self.wallet = RayonixWallet()
                if self.wallet.restore(str(wallet_file)):
                    # Set blockchain reference after successful restore
                    if self.wallet.set_blockchain_reference(self.rayonix_chain):
                        logger.info("Wallet loaded and blockchain integration established")
                    else:
                        logger.warning("Wallet loaded but blockchain integration failed")
                else:
                    logger.warning("Failed to load wallet from file, creating new one")
                    self._create_new_wallet_with_blockchain()
                    
            except Exception as e:
                logger.error(f"Error loading wallet: {e}")
                self._create_new_wallet_with_blockchain()
                
        else:
            self._create_new_wallet_with_blockchain()
    
    def _create_new_wallet_with_blockchain(self):
        """Create new wallet with blockchain integration"""
        try:
            from rayonix_wallet.core.wallet import create_new_wallet
            from rayonix_wallet.core.wallet_config import WalletType, AddressType
            
            # Create new wallet
            self.wallet = create_new_wallet(
                wallet_type=WalletType.HD,
                network=self.config_manager.get('network.network_type', 'testnet'),
                address_type=AddressType.RAYONIX
            )
            # Establish blockchain reference
            if self.wallet.set_blockchain_reference(self.rayonix_chain):
                logger.info("New wallet created with blockchain integration")
                
                # Save wallet immediately
                wallet_file = Path(self.config_manager.get('database.db_path', './rayonix_data')) / 'wallet.dat'
                if self.wallet.backup(str(wallet_file)):
                    logger.info("New wallet saved to disk")
                else:
                    logger.warning("Failed to save new wallet to disk")
            else:
                logger.error("Failed to establish blockchain integration for new wallet")
                self.wallet = None
        except Exception as e:
            logger.error(f"Failed to create new wallet: {e}")
            self.wallet = None
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        return self.config_manager.get(key, default)
    
    def _create_background_task(self, coro) -> asyncio.Task:
        """Create a background task and track it for proper cleanup"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task
    
    async def start(self):
        """Start the node"""
        if self.running:
            logger.warning("Node is already running")
            return False
        
        try:
            logger.info("Starting RAYONIX Node...")
            self.running = True
            self.shutdown_event.clear()
            
            # Start network if enabled
            if self.network:
                self._create_background_task(self.network.start())
            
            # Start background tasks
            self._create_background_task(self.sync_manager.sync_blocks())
            self._create_background_task(self.peer_monitor.monitor_peers())
            self._create_background_task(self.mempool_task.process_mempool())
            
            # Start staking if enabled
            if (self.get_config_value('consensus.consensus_type', 'pos') == 'pos' and 
                self.wallet):
                self._create_background_task(self.staking_task.staking_loop())
            
            # Start API server if enabled
            if self.api_server:
                if not await self.api_server.start():
                    logger.error("Failed to start API server")
            
            logger.info("RAYONIX Node started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start node: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def stop(self):
        """Stop the node gracefully with comprehensive cleanup"""
        if not self.running:
            return
        
        logger.info("Stopping RAYONIX Node...")
        self.running = False
        self.shutdown_event.set()
        
        # Cancel all background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if self.background_tasks:
            await asyncio.wait(self.background_tasks, timeout=10.0)
        
        # Stop API server
        if self.api_server:
            await self.api_server.stop()
        
        # Save wallet if loaded
        if self.wallet:
            wallet_file = Path(self.get_config_value('database.db_path', './rayonix_data')) / 'wallet.dat'
            self.wallet.backup(str(wallet_file))
            logger.info("Wallet saved")
        
        # Stop network
        if self.network:
            await self.network.stop()
        
        # Save state through rayonix_chain
        if self.rayonix_chain:
            self.rayonix_chain.close()
        
        logger.info("RAYONIX Node stopped gracefully")