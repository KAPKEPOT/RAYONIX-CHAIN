# blockchain/core/rayonix_chain.py
import os
import time
import asyncio
import logging
import signal
import threading
from typing import Dict, List, Any, Optional, Deque, Tuple, Set, Callable
from collections import defaultdict, deque
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pickle
import hashlib
import uuid

from blockchain.state.state_manager import StateManager
from blockchain.state.checkpoint_manager import CheckpointManager
from blockchain.validation.validation_manager import ValidationManager
from blockchain.transactions.transaction_manager import TransactionManager
from blockchain.fees.fee_estimator import FeeEstimator
from blockchain.forks.fork_manager import ForkManager
from blockchain.production.block_producer import BlockProducer
from blockchain.models.chain_state import ChainState
from rayonix_wallet.core.wallet import RayonixWallet
from utxo_system.database.core import UTXOSet
from consensusengine.core.consensus import ProofOfStake
from blockchain.utils.genesis import GenesisBlockGenerator
from network.core.p2p_network import AdvancedP2PNetwork
from blockchain.config.consensus_config import ConsensusConfig
from merkle_system.merkle import MerkleTree, CompactMerkleTree, MerkleTreeConfig, HashAlgorithm, ProofFormat, MerkleTreeFactory, MerkleTreeStats, global_stats, create_merkle_tree_from_file, create_merkle_tree_from_large_file, batch_verify_proofs, batch_verify_proofs_async, create_merkle_mountain_range

logger = logging.getLogger(__name__)

class BlockchainState(Enum):
    """Blockchain node state enumeration"""
    INITIALIZING = "initializing"
    STOPPED = "stopped"
    SYNCING = "syncing"
    SYNCED = "synced"
    FORKED = "forked"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

class NodeHealth(Enum):
    """Node health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class NodeMetrics:
    """Comprehensive node metrics container"""
    # Blockchain metrics
    block_height: int = 0
    total_blocks_processed: int = 0
    total_transactions_processed: int = 0
    average_block_processing_time: float = 0.0
    chain_reorganizations: int = 0
    
    # Network metrics
    connected_peers: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    peer_quality_score: float = 0.0
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_gb: float = 0.0
    database_size_gb: float = 0.0
    
    # Consensus metrics
    validator_status: str = "inactive"
    stake_amount: int = 0
    blocks_produced: int = 0
    consensus_participation: float = 0.0
    
    # System metrics
    uptime_seconds: float = 0.0
    last_block_time: float = 0.0
    sync_progress: float = 0.0
    node_health: NodeHealth = NodeHealth.HEALTHY

@dataclass
class BlockchainConfig:
    """Blockchain configuration container"""
    network_type: str = "mainnet"
    data_dir: str = "./rayonix_data"
    port: int = 30303
    max_connections: int = 50
    block_time_target: int = 30
    max_block_size: int = 4000000
    min_transaction_fee: int = 1
    stake_minimum: int = 1000
    developer_fee_percent: float = 0.05
    enable_auto_staking: bool = True
    enable_transaction_relay: bool = True
    enable_state_pruning: bool = True
    max_reorganization_depth: int = 100
    checkpoint_interval: int = 1000
    # Add genesis-related configuration fields
    genesis_premine: int = 1000000
    max_supply: int = 21000000
    block_reward: int = 50
    foundation_address: str = 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX'
    # Add other missing configuration fields
    genesis_description: str = 'Initial RAYONIX blockchain genesis block'
    consensus_algorithm: str = 'pos'
    security_level: str = 'high'

class RayonixBlockchain:
    """Production-ready RAYONIX blockchain engine with enterprise features"""
    
    def __init__(self, network_type: str = "mainnet", data_dir: str = "./rayonix_data", 
                 config: Optional[Dict[str, Any]] = None):
        self.network_type = network_type
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = BlockchainState.INITIALIZING
        self.health = NodeHealth.UNHEALTHY
        self.config = self._load_configuration(config)
        self.startup_time = time.time()
        
        # Initialize components with error handling
        self._initialize_components()
        
        # State management
        self.chain_head = None
        self.mempool_size = 0
        self.sync_progress = 0.0
        self.last_sync_update = 0.0
        
        # Performance tracking
        self.metrics = NodeMetrics()
        self.performance_history: Deque[NodeMetrics] = deque(maxlen=1000)
        self.error_counters = defaultdict(int)
        
        # Event subscribers
        self.event_subscribers: Dict[str, List[Callable]] = {
            'block_processed': [],
            'transaction_received': [],
            'chain_reorganization': [],
            'node_state_change': [],
            'consensus_participation': []
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Threading and synchronization
        self.lock = threading.RLock()
        self.chain_lock = asyncio.Lock()
        
        # Initialize blockchain
        self._initialize_blockchain()
        
        logger.info(f"RAYONIX node initialized for {network_type} network")
        
    def _config_to_dict(self) -> Dict[str, Any]:
    	"""Convert BlockchainConfig to dictionary"""
    	if hasattr(self.config, '__dataclass_fields__'):
    		return {field: getattr(self.config, field) for field in self.config.__dataclass_fields__}
    	elif isinstance(self.config, dict):
    		return self.config.copy()
    	else:
    		return vars(self.config) if hasattr(self.config, '__dict__') else {}        

    def _initialize_components(self):
        """Initialize all blockchain components with comprehensive error handling"""
        try:
            logger.info("Initializing blockchain components...")
            
            # Convert config to dict for components that need it
            config_dict = self._config_to_dict()
            
            # Initialize database with retry logic
            self.database = self._initialize_database_with_retry()
            
            # Initialize core components
            db_path = str(self.data_dir / 'utxo_db')
            self.utxo_set = UTXOSet(db_path)
            
            from consensusengine.utils.config.factory import ConfigFactory
            
            # Get configuration parameters from instance or use defaults
            consensus_params = getattr(self, 'consensus_config', {})
            network_params = getattr(self, 'network_config', {})
            
            logger.info(f"Consensus params: {list(consensus_params.keys())}")
            logger.info(f"Network params: {list(network_params.keys())}")
            
            # Create configurations with safe fallback
            consensus_config = ConfigFactory.create_safe_consensus_config(**consensus_params)
            network_config = ConfigFactory.create_network_config(**network_params)
            
            logger.info("Configurations created successfully")
            
            # Import consensus engine here to avoid circular imports
            from consensusengine.core.consensus import ProofOfStake
            self.consensus = ProofOfStake(
                config=consensus_config,
                network_config=network_config
            )
            logger.info("Consensus engine initialized successfully")
           
            
            self.contract_manager = self._initialize_contract_manager()
            if self.contract_manager is None:
            	raise RuntimeError("Contract manager initialization failed - cannot proceed without contract manager")
            	
            	
            self.wallet = self._initialize_wallet()
            
            # Initialize core managers with dependency injection
            self.state_manager = StateManager(
                self.database, self.utxo_set, self.consensus, self.contract_manager,
                state_path=str(self.data_dir / "state")
            )
            self._setup_contract_manager_references()
            
            self.checkpoint_manager = CheckpointManager(self.database, self.state_manager)
            self.validation_manager = ValidationManager(self.state_manager, config_dict)
            self.transaction_manager = TransactionManager(self.state_manager, self.wallet, config_dict)
            self.fee_estimator = FeeEstimator(self.state_manager, config_dict)
            self.fork_manager = ForkManager(self.state_manager, self.validation_manager, config_dict)
            self.block_producer = BlockProducer(self.state_manager, self.validation_manager, config_dict, self.wallet)
            
            # Initialize network layer
            self.network = AdvancedP2PNetwork(
                network_id=self._get_network_id(),
                port=self.config.port,
                max_connections=self.config.max_connections,
                node_id=self._generate_node_id()
            )
            
            self.health = NodeHealth.HEALTHY
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            self.health = NodeHealth.CRITICAL
            raise
            
    def _setup_contract_manager_references(self):
    	try:
    		if hasattr(self.contract_manager, 'set_blockchain_reference'):
    			self.contract_manager.set_blockchain_reference(self)
    		if hasattr(self.contract_manager, 'set_consensus_engine'):
    			self.contract_manager.set_consensus_engine(self.consensus)
    		if hasattr(self.contract_manager, 'set_state_manager'):
    			self.contract_manager.set_state_manager(self.state_manager)
    		if hasattr(self.contract_manager, 'initialize'):
    			self.contract_manager.initialize()
    		logger.info("Contract manager references set successfully")
    	except Exception as e:
    		logger.error(f"Failed to set contract manager references: {e}")
    		raise          

    def _initialize_database_with_retry(self, max_retries: int = 3) -> Any:
        """Initialize database with retry logic for production robustness"""
        for attempt in range(max_retries):
            try:
                db_path = self.data_dir / 'blockchain_db'
                db_path.mkdir(parents=True, exist_ok=True)
                
                # Use LevelDB for production (fallback to SQLite if not available)
                try:
                    import plyvel
                    raw_db = plyvel.DB(str(db_path), create_if_missing=True)
                    logger.info("LevelDB database initialized successfully")
                    
                    # Wrap with SafeDatabaseWrapper for consistent type handling
                    from blockchain.utils.database_wrapper import SafeDatabaseWrapper
                    db = SafeDatabaseWrapper(raw_db)
                    return db
                    
                except ImportError:
                    logger.warning("LevelDB not available, using SQLite fallback")
                    import sqlite3
                    db_file = db_path / "blockchain.sqlite"
                    raw_db = sqlite3.connect(str(db_file), check_same_thread=False)
                    raw_db.execute("PRAGMA journal_mode=WAL")
                    raw_db.execute("PRAGMA synchronous=NORMAL")
                    raw_db.execute("PRAGMA cache_size=-64000")  # 64MB cache
                    
                    # Wrap with SafeDatabaseWrapper
                    from blockchain.utils.database_wrapper import SafeDatabaseWrapper
                    db = SafeDatabaseWrapper(raw_db)
                    logger.info("SQLite database initialized successfully")
                    return db
                    
            except Exception as e:
                logger.error(f"Database initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("Failed to initialize database after retries")

    def _initialize_contract_manager(self) -> Any:
        """Initialize contract manager with enhanced features"""
        try:
            from smart_contract.core.contract_manager import ContractManager
            # Create contract database path
            contracts_db_path = self.data_dir / "contracts_db"
            contracts_db_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare contract manager configuration
            contract_config = {
                'db_path': str(contracts_db_path),
                'max_workers': getattr(self.config, 'contract_max_workers', 50),
                'gas_price_config': {
                    'min_gas_price': getattr(self.config, 'min_gas_price', 1),
                    'max_gas_price': getattr(self.config, 'max_gas_price', 100),
                    'base_gas_price': getattr(self.config, 'base_gas_price', 5),
                    'adjustment_sensitivity': 0.2,
                'update_interval': 30,
                'emergency_update_interval': 5,
                'max_mempool_size': 10000,
                'target_block_utilization': 0.7,
                }
            }
            # Initialize contract manager with proper configuration
            contract_manager = ContractManager(
                db_path=str(contracts_db_path),
                config=contract_config,
                max_workers=getattr(self.config, 'contract_max_workers', 50)
            )
            # Set up integration with blockchain components
           # contract_manager.set_blockchain_reference(self)
            #contract_manager.set_consensus_engine(self.consensus)
           # contract_manager.set_state_manager(self.state_manager)
            
            # Initialize contract manager
            #contract_manager.initialize()
            
            logger.info("Production contract manager initialized successfully")
            return contract_manager
            
        except ImportError as e:
        	logger.error(f"Contract manager import failed: {e}")
        	raise RuntimeError("Contract manager is required for production deployment")
        except Exception as e:
        	logger.error(f"Contract manager initialization failed: {e}")
        	raise RuntimeError(f"Contract manager initialization failed: {e}")
  
    def _initialize_wallet(self) -> RayonixWallet:
        """Initialize wallet with comprehensive key management"""
        try:
            from rayonix_wallet.core.config import WalletConfig
            # Create wallet data directory
            wallet_dir = self.data_dir / "wallet"
            wallet_dir.mkdir(parents=True, exist_ok=True)
            
            wallet_config = WalletConfig(
                network=self.network_type,          
                encryption=True,
                auto_backup=True,
                db_path=str(self.data_dir / "wallet" / "wallet.db"),
                address_type='rayonix',  # or appropriate default
                gap_limit=20,
                account_index=0
            )    
            wallet = RayonixWallet(wallet_config)
            
            # Initialize wallet if not exists
            if not wallet.is_initialized():
                logger.info("Initializing new wallet...")
                mnemonic = wallet.initialize_new_wallet()
                logger.info(f"New wallet created with mnemonic (first 10 chars): {mnemonic[:10]}...")
            
            # Generate initial addresses
            for i in range(5):
                wallet.derive_address(i)
            
            logger.info(f"Wallet initialized with {len(wallet.addresses)} addresses")
            return wallet
            
        except Exception as e:
            logger.error(f"Wallet initialization failed: {e}")
            raise
            
    def _initialize_state_manager(self):
    	"""Initialize state manager with safe database wrapper"""
    	try:
    		# Create safe database wrapper
    		from blockchain.utils.database_wrapper import SafeDatabaseWrapper
    		safe_db = SafeDatabaseWrapper(self.database)
    		# Initialize state manager with safe wrapper
    		self.state_manager = StateManager(
    		    safe_db,
    		    self.utxo_set,
    		    self.consensus,
    		    self.contract_manager,
    		    state_path=str(self.data_dir / "state")
    		)
    	except Exception as e:
    		logger.error(f"State manager initialization failed: {e}")
    		
    		# Fallback to basic initialization
    		self.state_manager = StateManager(
    		    self.database,
    		    self.utxo_set,
    		    self.consensus,
    		    self.contract_manager,
    		    state_path=str(self.data_dir / "state")
    		)            

    def _load_configuration(self, custom_config: Optional[Dict[str, Any]]) -> BlockchainConfig:
        """Load and validate node configuration"""
        # Default configuration for mainnet
        default_config = BlockchainConfig(
            network_type=self.network_type,
            data_dir=str(self.data_dir),
            port=30303 if self.network_type == "mainnet" else 30304,
            max_connections=50,
            block_time_target=30,
            max_block_size=4000000,
            min_transaction_fee=1,
            stake_minimum=1000,
            developer_fee_percent=0.05,
            enable_auto_staking=True,
            enable_transaction_relay=True,
            enable_state_pruning=True,
            max_reorganization_depth=100,
            checkpoint_interval=1000,
            genesis_premine=1000000,
            max_supply=21000000,
            block_reward=50,
            foundation_address='RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX',
            genesis_description='Initial RAYONIX blockchain genesis block',
            consensus_algorithm='pos',
            security_level='high'
        )
        
        # Apply custom configuration
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(default_config, key):
                    setattr(default_config, key, value)
        
        # Validate configuration
        self._validate_configuration(default_config)
        
        # Save configuration for reference
        config_file = self.data_dir / "node_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(default_config), f, indent=2)
        
        return default_config

    def _validate_configuration(self, config: BlockchainConfig):
        """Validate configuration parameters"""
        if config.stake_minimum < 100:
            raise ValueError("Stake minimum must be at least 100")
        
        if config.max_block_size > 10 * 1024 * 1024:  # 10MB max
            raise ValueError("Max block size too large")
        
        if config.developer_fee_percent > 0.1:  # 10% max
            raise ValueError("Developer fee percent too high")
        # Validate genesis-related parameters
        if config.genesis_premine <= 0:
        	raise ValueError("Premine amount must be positive")
        if config.max_supply <= config.genesis_premine:
        	raise ValueError("Max supply must be greater than premine amount")
        if config.block_reward <= 0:
        	raise ValueError("Block reward must be positive")
        if not config.foundation_address or len(config.foundation_address) < 10:
        	raise ValueError("Foundation address must be valid")

    def _get_network_id(self) -> int:
        """Get network ID based on network type"""
        network_ids = {
            "mainnet": 1,
            "testnet": 2,
            "devnet": 3
        }
        return network_ids.get(self.network_type, 0)

    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        node_id_file = self.data_dir / "node_id"
        if node_id_file.exists():
            with open(node_id_file, 'r') as f:
                return f.read().strip()
        
        new_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        with open(node_id_file, 'w') as f:
            f.write(new_id)
        return new_id

    def _initialize_blockchain(self):
        """Initialize or load blockchain state with comprehensive recovery"""
        try:
            with self.lock:
                # Check if blockchain exists
                if self._blockchain_exists():
                    self._load_blockchain_state()
                    logger.info(f"Blockchain loaded from storage (height: {self.metrics.block_height})")
                else:
                    self._create_genesis_blockchain()
                    logger.info("New blockchain created with genesis block")
                
                self.state = BlockchainState.STOPPED
                
        except Exception as e:
            logger.error(f"Blockchain initialization failed: {e}")
            raise RuntimeError(f"Blockchain initialization failed: {e}")
            

    def _blockchain_exists(self) -> bool:
        """Check if blockchain data exists"""
        try:
            chain_head = self.database.get(b'chain_head')
            return chain_head is not None
        except Exception:
            return False

    def _create_genesis_blockchain(self):
        """Create new blockchain with genesis block"""
        try:
            config_dict = asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else self.config
            
            generator = GenesisBlockGenerator(config_dict)
            
            # Use standard genesis generation with proper configuration
            genesis_config = {
                'premine_amount': self.config.genesis_premine,
                'foundation_address': self.config.foundation_address,
                'network_id': self._get_network_id(),
                'timestamp': int(time.time()),
                'difficulty': 1,
                'version': 1,
                'max_supply': self.config.max_supply,
                'block_time_target': self.config.block_time_target
            }
            genesis_block = generator.generate_genesis_block(genesis_config)
            
            if not genesis_block:
                raise ValueError("Genesis block generation returned None")
            
            # ENHANCED VALIDATION: Verify genesis block has correct height
            if genesis_block.header.height != 0:
            	logger.warning(f"Correcting genesis block height from {genesis_block.header.height} to 0")
            	
            	# Force correct height
            	genesis_block.header.height = 0
            	genesis_block.hash = genesis_block.header.calculate_hash()  # Recalculate hash

            # Validate genesis block structure
            if not self._validate_genesis_block(genesis_block):
                raise ValueError("Generated genesis block failed validation")
                
            # Store genesis block first
            self._store_block(genesis_block)
            
            # Set initial height in database
            #self.database.put(b'current_height', b'0')
            
            # Apply genesis block to state (this will set initial checksum)
            if not self.state_manager.apply_genesis_block(genesis_block):
            	raise ValueError("Failed to apply genesis block to state")
            
            # Apply genesis block to state
            if not self.state_manager.apply_genesis_block(genesis_block):
                raise ValueError("Failed to apply genesis block to state")
                
            # Update chain head
            self.chain_head = genesis_block.hash
            self.database.put(b'chain_head', genesis_block.hash.encode())
            
            # Create initial checkpoint
            self.checkpoint_manager.create_checkpoint(genesis_block)
            logger.info("Genesis blockchain created successfully at height 0")
            
        except Exception as e:
            logger.error(f"Genesis blockchain creation failed: {e}")
            if hasattr(e, '__cause__') and e.__cause__:
            	logger.error(f"Underlying cause: {e.__cause__}")
            raise RuntimeError(f"Failed to create genesis blockchain: {e}")
        		
    def _generate_standard_genesis(self, generator, config_dict) -> Any:
    	"""Standard genesis generation"""
    	return generator.generate_genesis_block()  
    	
    def _generate_minimal_genesis(self, generator, config_dict) -> Any:
    	"""Minimal genesis generation with safe parameters"""
    	minimal_config = {
    	    'premine_amount': 1000000,
    	    'foundation_address': 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX',
    	    'network_id': 1,
    	    'timestamp': int(time.time()),
    	    'difficulty': 1,
    	    'version': 1
    	}
    	return generator.generate_genesis_block(minimal_config)
       
   
    def _validate_genesis_block(self, genesis_block: Any) -> bool:
    	"""Special validation for genesis blocks"""
    	try:
    		# Basic structure checks
    		if genesis_block.header.height != 0:
    			logger.error(f"Invalid genesis block height: {genesis_block.header.height}. Must be 0.")
    			return False
    			
    		if genesis_block.header.previous_hash != '0' * 64:
    			logger.error(f"Invalid genesis previous hash: {genesis_block.header.previous_hash}. Must be 64 zeros.")
    			return False
    			
    		# Check for at least one transaction (the premine)
    		if len(genesis_block.transactions) == 0:
    			logger.error("Genesis block must have at least one transaction")
    			return False
    		# Genesis-specific validation
    		premine_tx = genesis_block.transactions[0]
    		if not hasattr(premine_tx, 'outputs') or len(premine_tx.outputs) == 0:
    			
    			return False
    			
    		# Validate premine amount matches configuration
    		expected_premine = self.config.genesis_premine
    		actual_premine = sum(output.amount for output in premine_tx.outputs)
    		if actual_premine != expected_premine:
    			logger.error(f"Premine amount mismatch: expected {expected_premine}, got {actual_premine}")
    			return False
    		
    		# Validate block hash integrity
    		calculated_hash = genesis_block.header.calculate_hash()
    		if genesis_block.hash != calculated_hash:
    			logger.error(f"Block hash mismatch: stored {genesis_block.hash}, calculated {calculated_hash}")
    			return False
    		
    		logger.info(f"Genesis block validation PASSED: height=1, hash={genesis_block.hash[:16]}...")
    		
    		return True
    		
    	except Exception:
    		logger.error(f"Genesis block validation exception: {e}")
    		
    		return False
    		
    def _create_fallback_genesis_blockchain(self) -> bool:
    	"""Create a minimal fallback genesis blockchain"""
    	try:
    		logger.warning("Creating fallback genesis blockchain")
    		# Create minimal genesis configuration
    		fallback_config = {
    		    'premine_amount': 1000000,
    		    'foundation_address': 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX',
    		    'network_id': 1,
    		    'timestamp': int(time.time()),
    		    'difficulty': 1,
    		    'version': 1
    		}
    		generator = GenesisBlockGenerator(fallback_config)
    		genesis_block = generator.generate_genesis_block()
    		# Apply with minimal validation
    		self.state_manager.apply_block(genesis_block)
    		self._store_block(genesis_block)
    		self.chain_head = genesis_block.hash
    		self.database.put(b'chain_head', genesis_block.hash.encode())
    		return True
    	except Exception as e:
    		logger.error(f"Fallback genesis creation also failed: {e}")
    		return False
    
    def _load_blockchain_state(self):
        """Load blockchain state from storage"""
        try:
            # Load chain head
            chain_head_bytes = self.database.get(b'chain_head')
            if not chain_head_bytes:
                raise ValueError("Chain head not found")
            if isinstance(chain_head_bytes, bytes):
            	self.chain_head = chain_head_bytes.decode()
            	
            else:
            	self.chain_head = chain_head_bytes
  
            #self.chain_head = chain_head_bytes.decode()
            
            # Load current height
            current_height = self.state_manager.get_current_height()
            self.metrics.block_height = current_height
            
            # Verify state integrity
            if not self.state_manager.verify_state_integrity():
                logger.warning("State integrity verification failed during load")
                # Use existing checkpoint recovery
                checkpoints = self.checkpoint_manager.list_checkpoints()
                if checkpoints:
                	latest = max(checkpoints, key=lambda x: x.height)
                	if self.checkpoint_manager.restore_from_checkpoint(latest.name):
                		logger.info("Recovered from checkpoint")
                	
                	else:
                		# Fall back to genesis
                		logger.warning("Checkpoint recovery failed, recreating genesis...")
                		self._create_genesis_blockchain()
            
            logger.info(f"Blockchain state loaded successfully (height: {current_height})")
            
        except Exception as e:
            logger.error(f"Blockchain state loading failed: {e}")
            raise

    def _attempt_blockchain_recovery(self) -> bool:
        """Attempt to recover corrupted blockchain state"""
        logger.warning("Attempting blockchain recovery...")
        
        try:
            # Try to restore from latest checkpoint
            config_dict = asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else self.config
            
            checkpoints = self.checkpoint_manager.list_checkpoints()
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.height)
                if self.checkpoint_manager.restore_from_checkpoint(latest_checkpoint.name):
                    logger.info(f"Recovered from checkpoint at height {latest_checkpoint['height']}")
                    return True
            
            # Fallback to genesis block
            genesis_block_data = self.database.get(b'genesis_block')
            if genesis_block_data:
                generator = GenesisBlockGenerator(config_dict)
                if self.state_manager.restore_checkpoint('genesis'):
                    logger.info("Recovered from genesis block")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Blockchain recovery attempt failed: {e}")
            return False

    async def start(self):
        """Start the blockchain node with comprehensive initialization"""
        if self.running:
            logger.warning("Node is already running")
            return
        
        logger.info("Starting RAYONIX blockchain node...")
        
        try:
            self.running = True
            self.shutdown_event.clear()
            self.state = BlockchainState.INITIALIZING
            
            # Start network layer
            await self.network.start()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Change state to syncing
            self.state = BlockchainState.SYNCING
            self._notify_subscribers('node_state_change', {'state': self.state})
            
            # Begin synchronization
            asyncio.create_task(self._synchronization_loop())
            
            logger.info("RAYONIX node started successfully")
            
        except Exception as e:
            logger.error(f"Node startup failed: {e}")
            self.health = NodeHealth.CRITICAL
            await self.stop()

    async def stop(self):
        """Stop the blockchain node gracefully"""
        if not self.running:
            return
        
        logger.info("Stopping RAYONIX blockchain node...")
        
        self.running = False
        self.shutdown_event.set()
        self.state = BlockchainState.STOPPED
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Stop network
        await self.network.stop()
        
        # Save state
        self._save_node_state()
        
        # Wait for tasks to complete
        try:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        except Exception as e:
            logger.debug(f"Background task cancellation: {e}")
        
        self.background_tasks.clear()
        logger.info("RAYONIX node stopped gracefully")

    async def _start_background_tasks(self):
        """Start all background maintenance tasks"""
        tasks = [
            ('block_production', self._block_production_loop, 1),
            ('mempool_management', self._mempool_management_loop, 60),
            ('state_pruning', self._state_pruning_loop, 3600),
            ('performance_monitoring', self._performance_monitoring_loop, 300),
            ('fork_monitoring', self._fork_monitoring_loop, 30),
            ('health_check', self._health_monitoring_loop, 60),
            ('metrics_collection', self._metrics_collection_loop, 30)
        ]
        
        for name, coro, interval in tasks:
            task = asyncio.create_task(self._managed_background_task(name, coro, interval))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _managed_background_task(self, name: str, coro: Callable, interval: float):
        """Run a background task with managed error handling"""
        while self.running and not self.shutdown_event.is_set():
            try:
                await coro()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background task {name} failed: {e}")
                self.error_counters[f"task_{name}"] += 1
                await asyncio.sleep(min(interval, 300))  # Cap backoff at 5 minutes

    async def _block_production_loop(self):
        """Proof-of-Stake block production loop"""
        if self.state != BlockchainState.SYNCED:
            return
        
        if not self.config.enable_auto_staking:
            return
        
        try:
            # Check if we should produce a block
            if await self.block_producer.should_produce_block():
                block = await self.block_producer.create_new_block()
                if block:
                    success = await self._process_new_block(block)
                    if success:
                        self._notify_subscribers('block_processed', {
                            'height': block.header.height,
                            'hash': block.hash,
                            'producer': self.wallet.get_primary_address()
                        })
        
        except Exception as e:
            logger.error(f"Block production error: {e}")
            self.error_counters['block_production'] += 1

    async def _mempool_management_loop(self):
        """Mempool management loop"""
        try:
            # Clean expired transactions
            expired_count = self.transaction_manager.clean_mempool()
            if expired_count > 0:
                logger.debug(f"Cleaned {expired_count} expired transactions from mempool")
            
            # Update fee estimates
            self.fee_estimator.update_estimates()
            
        except Exception as e:
            logger.error(f"Mempool management error: {e}")
            self.error_counters['mempool_management'] += 1

    async def _state_pruning_loop(self):
        """State pruning and compaction loop"""
        if not self.config.enable_state_pruning:
            return
        
        try:
            current_height = self.state_manager.get_current_height()
            prune_height = current_height - self.config.max_reorganization_depth
            
            if prune_height > 0:
                pruned_blocks = await self._prune_blocks_before(prune_height)
                if pruned_blocks > 0:
                    logger.info(f"Pruned {pruned_blocks} blocks before height {prune_height}")
            
            # Compact database periodically
            if current_height % 10000 == 0:  # Every 10,000 blocks
                await self._compact_database()
            
        except Exception as e:
            logger.error(f"State pruning error: {e}")
            self.error_counters['state_pruning'] += 1

    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        try:
            metrics = await self._collect_performance_metrics()
            self.performance_history.append(metrics)
            
            # Update health status based on metrics
            self._update_health_status(metrics)
            
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            self.error_counters['performance_monitoring'] += 1

    async def _fork_monitoring_loop(self):
        """Fork monitoring loop"""
        try:
            if self.state == BlockchainState.SYNCED:
                fork_risk = await self.fork_manager.assess_fork_risk()
                if fork_risk.risk_level == 'high':
                    logger.warning(f"High fork risk detected: {fork_risk}")
                    self.state = BlockchainState.FORKED
                    self._notify_subscribers('chain_reorganization', fork_risk)
        
        except Exception as e:
            logger.error(f"Fork monitoring error: {e}")
            self.error_counters['fork_monitoring'] += 1

    async def _health_monitoring_loop(self):
        """Health monitoring loop"""
        try:
            health_status = self._assess_health()
            if health_status != self.health:
                self.health = health_status
                logger.info(f"Node health changed to: {health_status.value}")
        
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
            self.error_counters['health_monitoring'] += 1

    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        try:
            self.metrics = await self._collect_comprehensive_metrics()
            self.metrics.node_health = self.health
            self.metrics.uptime_seconds = time.time() - self.startup_time
            
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            self.error_counters['metrics_collection'] += 1

    async def _synchronization_loop(self):
        """Blockchain synchronization loop"""
        try:
            # Connect to bootstrap nodes
            await self._connect_to_network()
            
            # Start synchronization
            sync_result = await self.network.synchronize_chain(self._process_received_block)
            
            if sync_result.success:
                self.state = BlockchainState.SYNCED
                self.sync_progress = 100.0
                logger.info("Blockchain synchronization completed")
            else:
                logger.error(f"Synchronization failed: {sync_result.error}")
                self.state = BlockchainState.STOPPED
        
        except Exception as e:
            logger.error(f"Synchronization error: {e}")
            self.state = BlockchainState.STOPPED

    async def _process_received_block(self, block: Any) -> bool:
        """Process a block received from the network"""
        async with self.chain_lock:
            try:
                # Validate block
                validation_result = await self.validation_manager.validate_block(block)
                if not validation_result.is_valid:
                    logger.warning(f"Block validation failed: {validation_result.errors}")
                    return False
                
                # Apply block to state
                if not await self.state_manager.apply_block(block):
                    logger.error("Failed to apply received block")
                    return False
                
                # Update chain head
                self.chain_head = block.hash
                self._store_block(block)
                
                # Update metrics
                self.metrics.total_blocks_processed += 1
                self.metrics.last_block_time = time.time()
                
                # Remove transactions from mempool
                self.transaction_manager.remove_from_mempool([tx.hash for tx in block.transactions])
                
                # Create checkpoint if needed
                self.checkpoint_manager.create_checkpoint_if_needed(block)
                
                logger.debug(f"Processed block #{block.header.height} - {block.hash[:16]}...")
                return True
                
            except Exception as e:
                logger.error(f"Block processing failed: {e}")
                return False

    async def _process_new_block(self, block: Any) -> bool:
        """Process a newly created block"""
        async with self.chain_lock:
            try:
                # Apply to state
                if not self.state_manager.apply_block(block):
                    logger.error("Failed to apply self-created block")
                    return False
                
                # Update chain head
                self.chain_head = block.hash
                self._store_block(block)
                
                # Broadcast to network
                await self.network.broadcast_block(block)
                
                # Remove transactions from mempool
                self.transaction_manager.remove_from_mempool([tx.hash for tx in block.transactions])
                
                logger.info(f"New block created: #{block.header.height} - {block.hash[:16]}...")
                return True
                
            except Exception as e:
                logger.error(f"New block processing failed: {e}")
                return False

    def _store_block(self, block: Any):
        """Store block in database"""
        try:
            block_data = pickle.dumps(block)
            self.database.put(block.hash.encode(), block_data)
            
            # Also store height index
            height_key = f"height_{block.header.height}".encode()
            self.database.put(height_key, block.hash.encode())
            
        except Exception as e:
            logger.error(f"Failed to store block: {e}")

    async def _prune_blocks_before(self, height: int) -> int:
        """Prune blocks before specified height"""
        if height <= 0:
        	logger.warning(f"Cannot prune blocks before height {height}, must be positive")
        	return 0
        
        current_height = self.state_manager.get_current_height()
        if height >= current_height:
        	logger.warning(f"Prune height {height} >= current height {current_height}, no blocks to prune")
        	return 0
        	
        # Safety check: don't prune too aggressively
        max_prune_depth = current_height - self.config.max_reorganization_depth
        if height > max_prune_depth:
        	logger.warning(f"Prune height {height} exceeds safe depth {max_prune_depth}, limiting")
        	height = max_prune_depth
        
        pruned_count = 0
        batch_size = 100
        
        try:
        	logger.info(f"Starting block pruning for blocks before height {height}")
        	
        	# Use checkpoint for safety
        	checkpoint_name = f"pre_prune_{height}"
        	self.checkpoint_manager.create_checkpoint(checkpoint_name)
        	
        	# Get blocks to prune in batches
        	for batch_start in range(0, height, batch_size):
        		batch_end = min(batch_start + batch_size, height)
        		
        		batch_pruned = await self._prune_block_batch(batch_start, batch_end)
        		pruned_count += batch_pruned
        		
        		# Small delay to prevent overwhelming the system
        		await asyncio.sleep(0.01)
        		
        		# Update metrics periodically
        		if batch_start % 1000 == 0:
        			logger.info(f"Pruned {pruned_count} blocks up to height {batch_start}")
        			
        	# Update UTXO set and state
        	await self._prune_utxo_set(height)
        	
        	# Update state manager
        	self.state_manager.update_pruned_height(height)
        	
        	# Create post-prune checkpoint
        	self.checkpoint_manager.create_checkpoint(f"post_prune_{height}")
        	
        	logger.info(f"Successfully pruned {pruned_count} blocks before height {height}")
        	
        	# Update metrics
        	with self.locks['stats']:
        		self.metrics.total_blocks_pruned += pruned_count
        	
        	return pruned_count
        	
        except Exception as e:
        	logger.error(f"Block pruning failed at height {height}: {e}")
        	
        	# Attempt recovery from checkpoint
        	try:
        		if self.checkpoint_manager.restore_from_checkpoint(checkpoint_name):
        			logger.info(f"Recovered from checkpoint {checkpoint_name} after pruning failure")
        			
        		else:
        			logger.error("Failed to recover from checkpoint after pruning failure")
        	except Exception as recovery_error:
        		logger.error(f"Checkpoint recovery also failed: {recovery_error}")
        	
        	raise DatabaseError(f"Block pruning failed: {e}")
        	
    async def _prune_block_batch(self, start_height: int, end_height: int) -> int:
    	"""Prune a batch of blocks"""
    	pruned_in_batch = 0
    	
    	try:
    		for height in range(start_height, end_height):
    			# Get block hash from height index
    			height_key = f"height_{height}".encode()
    			block_hash_bytes = self.database.get(height_key)
    			
    			if not block_hash_bytes:
    				continue
    				
    			block_hash = block_hash_bytes.decode() if isinstance(block_hash_bytes, bytes) else block_hash_bytes
    			
    			# Get block data
    			block_data = self.database.get(block_hash.encode())
    			
    			if not block_data:
    				continue
    				
    			# Deserialize block to get transaction information
    			try:
    				block = pickle.loads(block_data)
    				
    				# Remove block data
    				self.database.delete(block_hash.encode())
    				
    				# Remove height index
    				self.database.delete(height_key)
    				
    				# Remove transaction references if they exist
    				await self._remove_transaction_references(block)
    				
    				pruned_in_batch += 1
    				
    			except Exception as e:
    				logger.warning(f"Failed to process block at height {height}: {e}")
    				continue
    				
    		return pruned_in_batch
    	
    	except Exception as e:
    		logger.error(f"Batch pruning failed for heights {start_height}-{end_height}: {e}")
    		return 0
    		
    async def _prune_utxo_set(self, prune_height: int):
    	"""Prune UTXO set for blocks before specified height"""
    	try:
    		if hasattr(self.utxo_set, 'prune_before_height'):
    			pruned_utxos = self.utxo_set.prune_before_height(prune_height)
    			logger.info(f"Pruned {pruned_utxos} UTXOs before height {prune_height}")
    		
    		else:
    			logger.warning("UTXO set does not support pruning")
    	
    	except Exception as e:
    		logger.error(f"UTXO pruning failed: {e}")
    		# Continue with block pruning even if UTXO pruning fails
    		
    async def _remove_transaction_references(self, block: Any):
    	"""Remove transaction references from database"""
    	try:
    		if not hasattr(block, 'transactions'):
    			return
    			
    		for tx in block.transactions:
    			tx_hash = getattr(tx, 'hash', None)
    			if tx_hash:
    				# Remove transaction data if stored separately
    				tx_key = f"tx_{tx_hash}".encode()
    				self.database.delete(tx_key)
    				
    				# Remove from transaction index if it exists
    				tx_index_key = f"tx_index_{tx_hash}".encode()
    				
    				self.database.delete(tx_index_key)
    	
    	except Exception as e:
    		logger.warning(f"Failed to remove transaction references: {e}")

    async def _compact_database(self):
        """Compact database"""
        try:
            if hasattr(self.database, 'compact_range'):
                self.database.compact_range()
                logger.info("Database compaction completed")
        except Exception as e:
            logger.error(f"Database compaction failed: {e}")

    async def _connect_to_network(self):
        """Connect to network bootstrap nodes"""
        bootstrap_nodes = self._get_bootstrap_nodes()
        for node in bootstrap_nodes:
            try:
                await self.network.connect_to_node(node)
            except Exception as e:
                logger.debug(f"Failed to connect to bootstrap node {node}: {e}")

    def _get_bootstrap_nodes(self) -> List[str]:
        """Get bootstrap nodes for network"""
        if self.network_type == "mainnet":
            return [
                "node1.rayonix.site:30303",
                "node2.rayonix.site:30303",
                "node3.rayonix.site:30303"
            ]
        elif self.network_type == "testnet":
            return [
                "testnet-node1.rayonix.site:30304",
                "testnet-node2.rayonix.site:30304"
            ]
        else:
            return []

    def _assess_health(self) -> NodeHealth:
        """Assess node health based on various metrics"""
        error_thresholds = {
            'block_production': 10,
            'mempool_management': 5,
            'state_pruning': 3
        }
        
        # Check error counters
        for error_type, threshold in error_thresholds.items():
            if self.error_counters.get(error_type, 0) > threshold:
                return NodeHealth.UNHEALTHY
        
        # Check synchronization status
        if self.state == BlockchainState.SYNCING and time.time() - self.last_sync_update > 300:
            return NodeHealth.DEGRADED
        
        # Check memory usage
        if self.metrics.memory_usage_mb > 4096:  # 4GB threshold
            return NodeHealth.DEGRADED
        
        return NodeHealth.HEALTHY

    def _update_health_status(self, metrics: NodeMetrics):
        """Update health status based on metrics"""
        # Implementation would analyze various metrics to determine health
        pass

    async def _collect_comprehensive_metrics(self) -> NodeMetrics:
        """Collect comprehensive node metrics"""
        metrics = NodeMetrics()
        
        # Blockchain metrics
        metrics.block_height = self.state_manager.get_current_height()
        metrics.total_blocks_processed = self.metrics.total_blocks_processed
        metrics.sync_progress = self.sync_progress
        
        # Network metrics
        metrics.connected_peers = len(self.network.connected_peers)
        
        # Performance metrics (simplified)
        try:
            import psutil
            process = psutil.Process()
            metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            metrics.cpu_usage_percent = psutil.cpu_percent()
        except ImportError:
            pass
        
        return metrics

    def _save_node_state(self):
        """Save current node state to disk"""
        try:
            # Create checkpoint
            self.state_manager.create_checkpoint("shutdown_checkpoint")
            
            # Save node state
            state_data = {
                'chain_head': self.chain_head,
                'state': self.state.value,
                'health': self.health.value,
                'timestamp': time.time()
            }
            
            state_file = self.data_dir / "node_state.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f)
            
            logger.info("Node state saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save node state: {e}")

    def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
        """Notify event subscribers"""
        for callback in self.event_subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}")

    def subscribe_to_events(self, event_type: str, callback: Callable):
        """Subscribe to node events"""
        if event_type in self.event_subscribers:
            self.event_subscribers[event_type].append(callback)

    # Public API methods
    def get_balance(self, address: str) -> int:
        """Get balance for address"""
        return self.state_manager.utxo_set.get_balance(address)

    def get_transaction(self, tx_hash: str) -> Optional[Any]:
        """Get transaction by hash"""
        return self.transaction_manager.get_transaction(tx_hash)

    def get_block(self, identifier: Any) -> Optional[Any]:
        """Get block by height or hash"""
        try:
            if isinstance(identifier, int):
                # Get by height
                height_key = f"height_{identifier}".encode()
                block_hash = self.database.get(height_key)
                if block_hash:
                    identifier = block_hash.decode()
            
            # Get by hash
            block_data = self.database.get(identifier.encode())
            if block_data:
                return pickle.loads(block_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get block: {e}")
            return None

    def send_transaction(self, from_address: str, to_address: str, amount: int,
                       fee_strategy: str = 'medium', **kwargs) -> Any:
        """Send a transaction"""
        return self.transaction_manager.create_transaction(
            from_address, to_address, amount, fee_strategy, **kwargs
        )

    def get_blockchain_info(self) -> Dict[str, Any]:
        """Get comprehensive blockchain information"""
        return {
            'network': self.network_type,
            'height': self.metrics.block_height,
            'chain_head': self.chain_head,
            'state': self.state.value,
            'health': self.health.value,
            'sync_progress': self.sync_progress,
            'mempool_size': len(self.transaction_manager.mempool),
            'connected_peers': self.metrics.connected_peers,
            'uptime': self.metrics.uptime_seconds,
            'total_supply': self.consensus.get_total_supply(),
            'circulating_supply': self.consensus.get_circulating_supply()
        }

    def get_validator_info(self, address: Optional[str] = None) -> Dict[str, Any]:
        """Get validator information"""
        if not address and self.wallet.addresses:
            address = list(self.wallet.addresses.keys())[0]
        
        return self.consensus.get_validator_info(address) if address else {}

    def register_validator(self, stake_amount: int) -> bool:
        """Register as validator"""
        if stake_amount < self.config.stake_minimum:
            logger.error(f"Stake amount {stake_amount} below minimum {self.config.stake_minimum}")
            return False
        
        if not self.wallet.addresses:
            logger.error("No wallet addresses available for validation")
            return False
        
        validator_address = list(self.wallet.addresses.keys())[0]
        return self.consensus.register_validator(validator_address, stake_amount)

    def get_node_metrics(self) -> NodeMetrics:
        """Get current node metrics"""
        return self.metrics

    def get_performance_history(self) -> List[NodeMetrics]:
        """Get performance history"""
        return list(self.performance_history)

class StateRecoveryError(Exception):
    """State recovery failed"""
    pass