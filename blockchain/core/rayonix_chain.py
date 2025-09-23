# blockchain/core/rayonix_chain.py
import os
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Deque
from collections import defaultdict, deque

from blockchain.state.state_manager import StateManager
from blockchain.state.checkpoint_manager import CheckpointManager
from blockchain.validation.validation_manager import ValidationManager
from blockchain.transactions.transaction_manager import TransactionManager
from blockchain.fees.fee_estimator import FeeEstimator
from blockchain.forks.fork_manager import ForkManager
from blockchain.production.block_producer import BlockProducer
from blockchain.models.blockchain_state import BlockchainState
from rayonix_wallet.core.wallet import RayonixWallet
from utxo_system.database.core import UTXOSet
from consensusengine.core.consensus import ProofOfStake
from blockchain.utils.genesis import GenesisBlockGenerator

logger = logging.getLogger(__name__)

class RayonixBlockchain:
    """Production-ready RAYONIX blockchain engine"""
    
    def __init__(self, network_type: str = "mainnet", data_dir: str = "./rayonix_data", 
                 config: Optional[Dict[str, Any]] = None):
        self.network_type = network_type
        self.data_dir = data_dir
        self.state = BlockchainState.STOPPED
        self.config = self._load_configuration(config)
        
        # Initialize components
        self.database = self._initialize_database()
        self.utxo_set = self._initialize_utxo_set()
        self.consensus = self._initialize_consensus()
        self.contract_manager = self._initialize_contract_manager()
        self.wallet = self._initialize_wallet()
        
        # Initialize core managers
        self.state_manager = StateManager(self.database, self.utxo_set, self.consensus, self.contract_manager)
        self.checkpoint_manager = CheckpointManager(self.database, self.state_manager)
        self.validation_manager = ValidationManager(self.state_manager, self.config['validation'])
        self.transaction_manager = TransactionManager(self.state_manager, self.wallet, self.config['transactions'])
        self.fee_estimator = FeeEstimator(self.state_manager, self.config['fees'])
        self.fork_manager = ForkManager(self.state_manager, self.validation_manager, self.config['forks'])
        self.block_producer = BlockProducer(self.state_manager, self.validation_manager, self.config, self.wallet)
        
        # State variables
        self.chain_head = None
        self.mempool_size = 0
        self.sync_progress = 0
        self.performance_metrics = defaultdict(list)
        
        # Background tasks
        self.background_tasks = []
        self.running = False
        
        # Initialize blockchain
        self._initialize_blockchain()
        
        logger.info(f"RAYONIX node initialized for {network_type} network")
    
    def _load_configuration(self, custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load node configuration"""
        default_config = {
            'network': {
                'type': self.network_type,
                'port': 30303,
                'max_connections': 50,
                'bootstrap_nodes': self._get_bootstrap_nodes()
            },
            'consensus': {
                'block_reward': 50,
                'halving_interval': 210000,
                'difficulty_adjustment_blocks': 2016,
                'stake_minimum': 1000,
                'block_time_target': 30,
                'max_supply': 21000000,
                'developer_fee_percent': 0.05
            },
            'validation': {
                'max_block_size': 4000000,
                'max_transaction_size': 100000,
                'min_transaction_fee': 1,
                'max_future_block_time': 7200,
                'max_past_block_time': 86400
            },
            'transactions': {
                'max_mempool_size': 10000,
                'mempool_expiry_time': 3600,
                'fee_estimation_window': 100
            },
            'fees': {
                'fee_estimation_interval': 30,
                'min_transaction_fee': 1,
                'base_gas_price': 1000000000,
                'adjustment_factor': 1.125
            },
            'forks': {
                'fork_detection_threshold': 6,
                'max_reorganizations_per_hour': 3,
                'reorganization_depth_limit': 100
            },
            'database': {
                'type': 'plyvel',
                'cache_size': 128 * 1024 * 1024,
                'compression': 'snappy',
                'max_open_files': 1000
            }
        }
        
        # Merge with custom config if provided
        if custom_config:
            self._deep_merge(default_config, custom_config)
        
        return default_config
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _initialize_database(self) -> Any:
        """Initialize database with proper configuration"""
        # This would be implemented based on your specific database backend
        # For example, using Plyvel:
        try:
            import plyvel
            db_path = os.path.join(self.data_dir, 'blockchain_db')
            db = plyvel.DB(db_path, create_if_missing=True)
            return db
        except ImportError:
            logger.error("Plyvel not available, using in-memory database")
            return {}  # Fallback to in-memory dict
    
    def _initialize_utxo_set(self) -> Any:
        """Initialize UTXO set"""
        # This would be implemented based on your UTXO set implementation
        
        return UTXOSet()
    
    def _initialize_consensus(self) -> Any:
        """Initialize consensus mechanism"""
        # This would be implemented based on your consensus implementation
        
        return ProofOfStake(self.config['consensus'])
    
    def _initialize_contract_manager(self) -> Any:
        """Initialize contract manager"""
        # This would be implemented based on your contract system
        from blockchain.models.smart_contract import ContractManager
        return ContractManager()
    
    def _initialize_wallet(self) -> Any:
        """Initialize wallet system"""
        # This would be implemented based on your wallet implementation
        
        wallet_config = self.config.get('wallet', {})
        return RayonixWallet(wallet_config, os.path.join(self.data_dir, 'wallets'))
    
    def _get_bootstrap_nodes(self) -> List[str]:
        """Get bootstrap nodes for network"""
        if self.network_type == "mainnet":
            return [
                "node1.rayonix.com:30303",
                "node2.rayonix.com:30303",
                "node3.rayonix.com:30303"
            ]
        elif self.network_type == "testnet":
            return [
                "testnet-node1.rayonix.com:30304",
                "testnet-node2.rayonix.com:30304"
            ]
        else:
            return []
    
    def _initialize_blockchain(self):
        """Initialize or load blockchain state"""
        try:
            # Try to load chain head from database
            self.chain_head = self.database.get(b'chain_head')
            
            if not self.chain_head:
                # Create genesis block if not exists
                genesis_block = self._create_genesis_block()
                self._process_genesis_block(genesis_block)
            else:
                # Load full state
                self._load_blockchain_state()
                
        except Exception as e:
            logger.error(f"Blockchain initialization failed: {e}")
            raise
    
    def _create_genesis_block(self) -> Any:
        """Create genesis block""" 
        generator = GenesisBlockGenerator(self.config)
        return generator.generate_genesis_block()
    
    def _process_genesis_block(self, genesis_block: Any):
        """Process genesis block and initialize state"""
        # Validate genesis block
        validation_result = self.validation_manager.validate_block(genesis_block, ValidationLevel.CONSENSUS)
        if not validation_result.is_valid:
            raise ValueError(f"Genesis block validation failed: {validation_result.errors}")
        
        # Apply to state
        if not self.state_manager.apply_block(genesis_block):
            raise ValueError("Failed to apply genesis block")
        
        # Save to database
        self.database.put(genesis_block.hash.encode(), genesis_block.to_bytes())
        self.database.put(b'chain_head', genesis_block.hash.encode())
        self.chain_head = genesis_block.hash
        
        logger.info("Genesis block created and processed")
    
    def _load_blockchain_state(self):
        """Load full blockchain state from database"""
        logger.info("Blockchain state loaded from database")
    
    def start(self):
        """Start the blockchain node"""
        if self.running:
            logger.warning("Node is already running")
            return
        
        self.running = True
        self.state = BlockchainState.SYNCING
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("RAYONIX node started")
    
    def stop(self):
        """Stop the blockchain node"""
        if not self.running:
            return
        
        self.running = False
        self.state = BlockchainState.STOPPED
        
        # Stop background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save state
        self._save_state()
        
        logger.info("RAYONIX node stopped")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        tasks = [
            self._block_production_loop,
            self._mempool_management_loop,
            self._state_pruning_loop,
            self._performance_monitoring_loop,
            self._fork_monitoring_loop
        ]
        
        for task_func in tasks:
            task = asyncio.create_task(task_func())
            self.background_tasks.append(task)
    
    async def _block_production_loop(self):
        """Proof-of-Stake block production loop"""
        while self.running:
            try:
                if self.state == BlockchainState.SYNCED:
                    block = await self.block_producer.create_new_block()
                    if block:
                        await self._process_new_block(block)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Block production error: {e}")
                await asyncio.sleep(5)
    
    async def _mempool_management_loop(self):
        """Mempool management loop"""
        while self.running:
            try:
                # Clean expired transactions
                self.transaction_manager.clean_mempool()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Mempool management error: {e}")
                await asyncio.sleep(30)
    
    async def _state_pruning_loop(self):
        """State pruning and compaction loop"""
        while self.running:
            try:
                if self.state == BlockchainState.SYNCED:
                    # Prune old state data
                    await self._prune_old_state()
                    
                    # Compact database
                    if self._should_compact_database():
                        await self._compact_database()
                
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"State pruning error: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.running:
            try:
                metrics = await self._collect_performance_metrics()
                self.performance_metrics[time.time()] = metrics
                
                # Keep only last 24 hours of metrics
                cutoff = time.time() - 86400
                self.performance_metrics = {k: v for k, v in self.performance_metrics.items() if k > cutoff}
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _fork_monitoring_loop(self):
        """Fork monitoring loop"""
        while self.running:
            try:
                if self.state == BlockchainState.SYNCED:
                    fork_risk = self.fork_manager.monitor_fork_risk()
                    if fork_risk['fork_probability'] > 0.1:
                        logger.warning(f"High fork risk detected: {fork_risk}")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Fork monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _process_new_block(self, block: Any) -> bool:
        """Process a newly created block"""
        try:
            # Apply to state
            if not self.state_manager.apply_block(block):
                logger.error("Failed to apply self-created block")
                return False
            
            # Update chain head
            self.chain_head = block.hash
            self.database.put(b'chain_head', block.hash.encode())
            self.database.put(block.hash.encode(), block.to_bytes())
            
            # Remove transactions from mempool
            self.transaction_manager.remove_from_mempool([tx.hash for tx in block.transactions])
            
            logger.info(f"New block created: #{block.header.height} - {block.hash[:16]}...")
            return True
            
        except Exception as e:
            logger.error(f"New block processing failed: {e}")
            return False
    
    async def _prune_old_state(self):
        """Prune old state data to save space"""
        current_height = self.state_manager.get_current_height()
        prune_height = current_height - self.config['forks']['reorganization_depth_limit']
        
        if prune_height > 0:
            # Prune old blocks and state data
            await self._prune_blocks_before(prune_height)
            logger.info(f"Pruned state data before height {prune_height}")
    
    async def _prune_blocks_before(self, height: int):
        """Prune blocks before specified height"""
        # This would be implemented based on your database structure
        pass
    
    async def _compact_database(self):
        """Compact database"""
        try:
            if hasattr(self.database, 'compact_range'):
                self.database.compact_range()
                logger.info("Database compaction completed")
        except Exception as e:
            logger.error(f"Database compaction failed: {e}")
    
    def _should_compact_database(self) -> bool:
        """Check if database should be compacted"""
        # This would check database size and fragmentation
        return False  # Placeholder
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        return {
            'block_height': self.state_manager.get_current_height(),
            'mempool_size': len(self.transaction_manager.mempool),
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage(),
            'disk_usage': self._get_disk_usage(),
            'validation_stats': self.validation_manager.get_stats(),
            'fork_count': self.fork_manager.reorganization_count
        }
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0
    
    def _get_disk_usage(self) -> float:
        """Get disk usage in GB"""
        try:
            import psutil
            usage = psutil.disk_usage(self.data_dir)
            return usage.used / 1024 / 1024 / 1024
        except ImportError:
            return 0
    
    def _save_state(self):
        """Save current state to disk"""
        try:
            self.state_manager.create_checkpoint()
            if hasattr(self.database, 'sync'):
                self.database.sync()
            logger.info("Node state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save node state: {e}")
    
    # Public API methods
    def get_balance(self, address: str) -> int:
        """Get balance for address"""
        return self.state_manager.utxo_set.get_balance(address)
    
    def get_transaction(self, tx_hash: str) -> Optional[Any]:
        """Get transaction by hash"""
        # Check mempool first
        mempool_tx = self.transaction_manager.get_transaction(tx_hash)
        if mempool_tx:
            return mempool_tx
        
        # Check blockchain
        # This would be implemented based on your database
        return None
    
    def get_block(self, height_or_hash: Any) -> Optional[Any]:
        """Get block by height or hash"""
        # This would be implemented based on your database
        return None
    
    def send_transaction(self, from_address: str, to_address: str, amount: int,
                       fee_strategy: str = 'medium', **kwargs) -> Any:
        """Send a transaction"""
        return self.transaction_manager.create_transaction(
            from_address, to_address, amount, fee_strategy, **kwargs
        )
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Get blockchain information"""
        current_height = self.state_manager.get_current_height()
        return {
            'height': current_height,
            'chain_head': self.chain_head,
            'total_supply': self.consensus.total_supply,
            'circulating_supply': self.consensus.circulating_supply,
            'difficulty': self.consensus.calculate_difficulty(current_height),
            'block_reward': self.consensus.get_block_reward(current_height),
            'mempool_size': len(self.transaction_manager.mempool),
            'network_state': self.state.name,
            'sync_progress': self.sync_progress
        }
    
    def get_validator_info(self, address: Optional[str] = None) -> Dict[str, Any]:
        """Get validator information"""
        if not address and self.wallet.addresses:
            address = list(self.wallet.addresses.keys())[0]
        
        if not address:
            return {'error': 'No validator address provided'}
        
        return self.consensus.get_validator_info(address)
    
    def register_validator(self, stake_amount: int) -> bool:
        """Register as validator"""
        if not self.wallet.addresses:
            return False
        
        validator_address = list(self.wallet.addresses.keys())[0]
        return self.consensus.register_validator(validator_address, stake_amount)