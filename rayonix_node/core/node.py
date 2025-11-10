# rayonix_node/core/node.py - Main RayonixNode class

import asyncio
import threading
from typing import Optional, Set, Any  
from pathlib import Path
import logging
import time

from rayonix_node.core.dependencies import NodeDependencies
from rayonix_node.core.state_manager import NodeStateManager
from rayonix_node.network.network_manager import NetworkManager
from rayonix_node.api.server import RayonixAPIServer
from rayonix_node.tasks.staking_task import StakingTask
from rayonix_node.tasks.mempool_task import MempoolTask
from rayonix_node.tasks.peer_monitor import PeerMonitor
from rayonix_node.network.sync_manager import SyncManager
from rayonix_wallet.core.wallet_types import WalletType, AddressType

logger = logging.getLogger("rayonix_node.core")

class RayonixNode:
    def __init__(self, dependencies: Optional[NodeDependencies] = None):
        # Use provided dependencies or create empty ones
        if dependencies:
            self.deps = dependencies
            self.config_manager = dependencies.config_manager
            self.rayonix_chain = dependencies.rayonix_chain
            self.wallet = dependencies.wallet
            self.network = dependencies.network
            self.contract_manager = dependencies.contract_manager
            self.database = dependencies.database
            self.state_manager = NodeStateManager()
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
                from config.config_manager import ConfigManager
                self.config_manager = ConfigManager(config_path, encryption_key, auto_reload=True)
            
            # Initialize rayonix_chain if not provided via dependencies
            if not self.rayonix_chain:
                from blockchain.core.rayonix_chain import RayonixBlockchain
                
                network_type = self.config_manager.get('network.network_type')
                data_dir = Path(self.config_manager.get('database.db_path'))
                data_dir.mkdir(exist_ok=True, parents=True)
                # Create blockchain configuration from main config
                blockchain_config = {
                    'network_type': network_type,
                    'data_dir': str(data_dir),
                    'port': self.config_manager.get('network.listen_port'),
                    'max_connections': self.config_manager.get('network.max_connections'),
                    'block_time_target': self.config_manager.get('consensus.block_time'),
                    'max_block_size': self.config_manager.get('consensus.max_block_size'),
                    'min_transaction_fee': self.config_manager.get('gas.min_transaction_fee'),
                    'stake_minimum': self.config_manager.get('consensus.min_stake'),
                    'developer_fee_percent': self.config_manager.get('consensus.developer_fee_percent'),
                    'enable_auto_staking': self.config_manager.get('consensus.enable_auto_staking'),
                    'enable_transaction_relay': self.config_manager.get('network.enable_transaction_relay'),
                    'enable_state_pruning': self.config_manager.get('database.enable_state_pruning'),
                    'max_reorganization_depth': self.config_manager.get('consensus.max_reorganization_depth'),
                    'checkpoint_interval': self.config_manager.get('database.checkpoint_interval')
                }

                self.rayonix_chain = RayonixBlockchain(
                    network_type=network_type,
                    data_dir=str(data_dir),
                    config=blockchain_config
                )
                # Set node reference on the blockchain
                self.rayonix_chain.node = self
            
            # Set node reference on the blockchain
            #if not self.wallet:
                #await self._initialize_wallet_with_blockchain()
            self.wallet = None
           
            
            # Initialize network if enabled and not provided via dependencies
            if self.config_manager.get('network.enabled', True) and not self.network:
                await self.network_manager.initialize_network()
                self.network = self.network_manager.network
            
            # Initialize API server if enabled
            if self.config_manager.get('api.enabled'):
                api_host = self.config_manager.get('api.host')
                api_port = self.config_manager.get('api.port')
                self.api_server = RayonixAPIServer(self, api_host, api_port)
            
            logger.info("RAYONIX Node components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize node components: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def create_wallet_if_not_exists(self) -> bool:
    	"""Create wallet only when explicitly needed"""
    	if self.wallet is not None:
    		return True  # Wallet already exists
    	wallet_file = Path(self.get_config_value('database.db_path', './rayonix_data')) / 'wallet.dat'
    	
    	if wallet_file.exists() and wallet_file.stat().st_size > 0:
    		# Try to load existing wallet
    		return await self._initialize_wallet_with_blockchain()
    		
    	else:
    		# Create new wallet on demand
    		return await self._create_wallet_on_demand()
    		
    async def _create_wallet_on_demand(self) -> bool:
    	"""Create wallet when explicitly requested by user"""
    	print("=== DEBUG: _create_wallet_on_demand STARTED ===")
    	try:
    		print("DEBUG: Step 1 - Checking if rayonix_wallet package exists...")
    		import rayonix_wallet
    		print(f"DEBUG: rayonix_wallet package found: {rayonix_wallet.__file__}")
    		
    		print("DEBUG: Step 2 - Attempting to import WalletFactory...")
    		from config.config_manager import ConfigManager
    		from rayonix_wallet.core.wallet_factory import WalletFactory
    		print("DEBUG: ✅ WalletFactory imported successfully")  	
    			
    		print("DEBUG: Step 3 - Attempting to import WalletType and AddressType...")
    		from rayonix_wallet.core.wallet_types import WalletType, AddressType
    		print("DEBUG: ✅ WalletType and AddressType imported successfully")
    		
    		print("DEBUG: Step 4 - Creating wallet with factory...")    		
    		wallet, mnemonic = WalletFactory.create_new_wallet(
    		    wallet_type=WalletType.HD,
    		    address_type=AddressType.RAYONIX,
    		    config_manager=self.config_manager
    		)
    		
    		print("DEBUG: ✅ Wallet created successfully")
    		
    		self.wallet = wallet
    		self.wallet.creation_mnemonic = mnemonic
    		current_network = self.config_manager.config.network.network_type
    		print(f"DEBUG: Wallet assigned to node.wallet: {self.wallet is not None}")
    		
    		# Save wallet to file
    		wallet_file = Path(self.get_config_value('database.db_path', './rayonix_data')) / 'wallet.dat'
    		print(f"DEBUG: Attempting to save wallet to: {wallet_file}")
    		
    		try:
    			if hasattr(wallet, 'backup'):
    				wallet.backup(str(wallet_file))
    				print("DEBUG: ✅ Wallet backup attempted")
    			else:
    				print("DEBUG: ⚠️  Wallet backup method not available")
    		except Exception as backup_error:
    			print(f"DEBUG: ⚠️  Wallet backup failed: {backup_error}")
    		print("=== DEBUG: _create_wallet_on_demand COMPLETED SUCCESSFULLY ===")
    			
    		return True
    	
    	except ImportError as e:
    		print(f"DEBUG: ❌ IMPORT ERROR: {e}")
    		print(f"DEBUG: ImportError details: {type(e).__name__}: {e}")
    		import traceback
    		traceback.print_exc()
    		return False
    	
    	except Exception as e:
    		print(f"DEBUG: ❌ GENERAL ERROR: {e}")
    		print(f"DEBUG: Error details: {type(e).__name__}: {e}")
    		import traceback
    		traceback.print_exc()
    		return False

    async def _initialize_wallet_with_blockchain(self):
        """Initialize wallet with proper blockchain reference integration"""
        if not self.wallet:
        	return False
        	
        try:
        	# Method 1: Use set_blockchain_reference if available
        	if hasattr(self.wallet, 'set_blockchain_reference'):
        		return self.wallet.set_blockchain_reference(self.rayonix_chain)
        	
        	# Method 2: Set directly if attribute exists
        	elif hasattr(self.wallet, 'rayonix_chain'):
        		self.wallet.rayonix_chain = self.rayonix_chain
        		return True
        	
        	# Method 3: Set via balance calculator
        	elif (hasattr(self.wallet, 'balance_calculator') and hasattr(self.wallet.balance_calculator, 'rayonix_chain')):
        	     self.wallet.balance_calculator.rayonix_chain = self.rayonix_chain
        	     return True
        	logger.warning("No blockchain reference method found for wallet")
        	return False
        	     
        except Exception as e:
        	logger.error(f"Failed to set blockchain reference: {e}")
        	return False  
    def _create_new_wallet_with_factory(self, wallet_file: Path, network_type: str):
    	"""Create new wallet using factory pattern"""
    	try:
    		from rayonix_wallet.core.wallet_factory import WalletFactory
    		from rayonix_wallet.core.types import WalletType, AddressType
    		
    		self.wallet, mnemonic = WalletFactory.create_new_wallet(
    		    
    		    wallet_type=WalletType.HD,
    		    network=network_type,
    		    address_type=AddressType.RAYONIX
    		)
    		
    		logger.info("New wallet created successfully")
    		
    		# Unlock wallet before backup
    		if hasattr(self.wallet, 'locked') and self.wallet.locked:
    			self.wallet.unlock("", timeout=300)
    			
    		# Save the new wallet
    		if self.wallet.backup(str(wallet_file)):
    			logger.info(f"New wallet saved to {wallet_file}")
    			logger.info(f"New wallet mnemonic: {mnemonic}")
    			
    		else:
    			logger.warning("Failed to save new wallet to disk")
    			
    	except Exception as e:
    		logger.error(f"Failed to create new wallet: {e}")
    		

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
            
    async def load_wallet_on_demand(self, mnemonic: str, password: str = None) -> bool:
    	"""Load existing wallet from mnemonic"""
    	try:
    		from rayonix_wallet.core.wallet_factory import WalletFactory
    		wallet = WalletFactory.create_wallet_from_mnemonic(
    		    mnemonic=mnemonic,
    		    passphrase=password or "",
    		    wallet_type=WalletType.HD,
    		    address_type=AddressType.RAYONIX,
    		    config_manager=self.config_manager
    		)
    		
    		if wallet:
    			self.wallet = wallet
    			# Establish blockchain reference
    			if hasattr(wallet, 'set_blockchain_reference'):
    				wallet.set_blockchain_reference(self.rayonix_chain)
    			return True
    		return False
    		
    	except Exception as e:
    		logger.error(f"Failed to load wallet: {e}")
    		return False
    		
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        return self.config_manager.get(key, default)
    
    def _create_background_task(self, coro) -> asyncio.Task:
        """Create a background task and track it for proper cleanup"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task
    
    async def _verify_network_compatibility(self):
        """Verify that network implementation has all required methods"""
        required_methods = [
            'start', 'stop', 'get_peers', 'get_connected_peers', 
            'get_stats', 'connect_to_peer', 'disconnect_peer',
            'send_message', 'broadcast_message'
        ]
        
        if not self.network:
            logger.warning("No network instance available for compatibility check")
            return False
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(self.network, method) or not callable(getattr(self.network, method)):
                missing_methods.append(method)
        
        if missing_methods:
            logger.error(f"Network implementation missing required methods: {missing_methods}")
            return False
        
        logger.info("Network compatibility check passed")
        return True
    
    async def persist_wallet_state(self):
    	"""Persist wallet state to disk"""
    	if not self.wallet:
    		return False
    	try:
    		wallet_file = Path(self.get_config_value('database.db_path', './rayonix_data')) / 'wallet.dat'
    		
    		if hasattr(self.wallet, 'backup'):
    			return self.wallet.backup(str(wallet_file))
    		else:
    			logger.warning("Wallet backup method not available")
    			return False
    			
    	except Exception as e:
    		logger.error(f"Failed to persist wallet state: {e}")
    		return False
    		
    async def start(self):
        """Start the node"""
        if self.running:
            logger.warning("Node is already running")
            return False
        
        try:
            logger.info("Starting RAYONIX Node...")
            
            # Verify network compatibility before starting
            if self.network and not await self._verify_network_compatibility():
                logger.error("Network compatibility check failed. Node cannot start.")
                return False
            
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