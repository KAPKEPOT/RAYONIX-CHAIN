# rayonix_coin.py
import os
import hashlib
import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import ecdsa
from ecdsa import SECP256k1, SigningKey, VerifyingKey
import base58
import bech32
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import plyvel
from merkle import MerkleTree, CompactMerkleTree
from utxo import UTXOSet, Transaction, UTXO
from consensus import ProofOfStake, Validator
from smart_contract import ContractManager, SmartContract
from database import AdvancedDatabase
from wallet import RayonixWallet, WalletConfig, WalletType, AddressType, create_new_wallet
from p2p_network import AdvancedP2PNetwork, NodeConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RayonixCoin:
    
    def __init__(self, network_type: str = "mainnet", data_dir: str = "./rayonix_data"):
        self.network_type = network_type
        self.data_dir = data_dir
        self.genesis_block = None
        self.blockchain = []
        self.mempool = []
        self.utxo_set = UTXOSet()
        self.consensus = ProofOfStake()
        self.contract_manager = ContractManager()
        self.wallet = None
        self.network = None
        
        # Initialize database with plyvel
        self.database = AdvancedDatabase(f"{data_dir}/blockchain_db")

        # Configuration
        self.config = {
            'block_reward': 50,
            'halving_interval': 210000,
            'difficulty_adjustment_blocks': 2016,
            'max_block_size': 4000000,
            'max_transaction_size': 100000,
            'min_transaction_fee': 1,
            'stake_minimum': 1000,
            'block_time_target': 30,
            'max_supply': 21000000,
            'premine_amount': 1000000,
            'foundation_address': 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX',
            'developer_fee_percent': 0.05,
            'network_id': self._get_network_id(network_type)
        }
        
        # State
        self.current_difficulty = 4
        self.total_supply = 0
        self.circulating_supply = 0
        self.staking_rewards_distributed = 0
        self.foundation_funds = 0
        self.last_block_time = time.time()
        
        # Initialize components
        self._initialize_blockchain()
        self._initialize_wallet()
        self._initialize_network()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _get_network_id(self, network_type: str) -> int:
        """Get network ID based on network type"""
        network_ids = {
            "mainnet": 1,
            "testnet": 2,
            "devnet": 3,
            "regtest": 4
        }
        return network_ids.get(network_type, 1)
    
    def _initialize_blockchain(self):
        """Initialize or load blockchain"""
        # Try to load from database
        if self.database.get('genesis_block'):
            self._load_blockchain()
        else:
            self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create genesis block with premine"""
        genesis_transactions = []
        
        # Premine transaction
        premine_tx = Transaction(
            inputs=[],
            outputs=[{
                'address': self.config['foundation_address'],
                'amount': self.config['premine_amount'],
                'locktime': 0
            }],
            locktime=0
        )
        genesis_transactions.append(premine_tx.to_dict())  # Convert to dict here
        
        # Create genesis block
        self.genesis_block = {
            'height': 0,
            'hash': '0' * 64,
            'previous_hash': '0' * 64,
            'merkle_root': self._calculate_merkle_root(genesis_transactions),
            'timestamp': int(time.time()),
            'difficulty': 1,
            'nonce': 0,
            'validator': 'genesis',
            'transactions': genesis_transactions,
            'version': 1,
            'chainwork': 1
        }
        
        # Calculate actual hash
        self.genesis_block['hash'] = self._calculate_block_hash(self.genesis_block)
        
        # Add to blockchain
        self.blockchain.append(self.genesis_block)
        
        # Update UTXO set
        self._update_utxo_set(self.genesis_block)
        
        # Update supply
        self.total_supply += self.config['premine_amount']
        self.circulating_supply += self.config['premine_amount']
        self.foundation_funds += self.config['premine_amount']
        
        # Save to database
        self._save_blockchain()
    
    def _load_blockchain(self):
        """Load blockchain from database"""
        try:
            self.genesis_block = self.database.get('genesis_block')
            chain_data = self.database.get('blockchain')
            
            if chain_data:
                self.blockchain = chain_data
                # Rebuild UTXO set by processing all blocks
                for block in self.blockchain:
                    self._update_utxo_set(block)
                
                # Calculate current supply
                self._calculate_supply()
                
            logger.info(f"Blockchain loaded with {len(self.blockchain)} blocks")
            
        except Exception as e:
            logger.error(f"Failed to load blockchain: {e}")
            self._create_genesis_block()
    
    def _save_blockchain(self):
        """Save blockchain to database"""
        try:
            self.database.put('genesis_block', self.genesis_block)
            self.database.put('blockchain', self.blockchain)
            self.database.put('utxo_set', self.utxo_set.to_dict())
            self.database.put('supply_info', {
                'total_supply': self.total_supply,
                'circulating_supply': self.circulating_supply,
                'staking_rewards': self.staking_rewards_distributed,
                'foundation_funds': self.foundation_funds
            })
        except Exception as e:
            logger.error(f"Failed to save blockchain: {e}")
    
    def _initialize_wallet(self):
        """Initialize wallet system"""
        wallet_config = WalletConfig(
        network=self.network_type,
        address_type=AddressType.RAYONIX,  # Use the enum directly
        encryption=True
    )
        self.wallet = RayonixWallet(wallet_config)
        
        # Check if wallet needs to be initialized (no master key)
        if not self.wallet.master_key:
        	# Create a new HD wallet
        	try:
        		mnemonic_phrase, xpub = self.wallet.create_hd_wallet()
        		logger.info(f"New HD wallet created with mnemonic: {mnemonic_phrase[:10]}...")
        	except Exception as e:
        	    logger.error(f"Failed to create HD wallet: {e}")
        	    
        	    # Fallback: create from random private key
        	    private_key = os.urandom(32).hex()
        	    self.wallet.create_from_private_key(private_key, WalletType.NON_HD)
        	    logger.info("Non-HD wallet created from random private key")
        	    
        # Generate initial addresses
        if self.wallet.master_key:
            for i in range(5):
                self.wallet.derive_address(i, False) 	    
      			
    
    def _initialize_network(self):
        """Initialize P2P network"""
        network_config = NodeConfig(
            network_type=self.network_type.upper(),
            listen_port=30303,
            max_connections=50,
            bootstrap_nodes=self._get_bootstrap_nodes()
        )
        self.network = AdvancedP2PNetwork(network_config)
        
        # Register message handlers
        self.network.register_message_handler('block', self._handle_block_message)
        self.network.register_message_handler('transaction', self._handle_transaction_message)
        self.network.register_message_handler('consensus', self._handle_consensus_message)
    
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
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Mining/staking thread
        self.mining_thread = threading.Thread(target=self._mining_loop, daemon=True)
        self.mining_thread.start()
        
        # Network sync thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
        # Mempool management thread
        self.mempool_thread = threading.Thread(target=self._mempool_loop, daemon=True)
        self.mempool_thread.start()
    
    def _mining_loop(self):
        """Proof-of-Stake mining loop"""
        while True:
            try:
                if self._should_mine_block():
                    new_block = self._create_new_block()
                    if new_block:
                        self._add_block(new_block)
                        self._broadcast_block(new_block)
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Mining error: {e}")
                time.sleep(5)
    
    def _should_mine_block(self) -> bool:
        """Check if we should mine a new block"""
        # Check if we're a validator with sufficient stake
        if self.wallet and self.wallet.addresses:
            validator_address = list(self.wallet.addresses.keys())[0]
            if self.consensus.is_validator(validator_address):
            	current_time = time.time()
            	# Check if it's our turn to validate
            	return self.consensus.should_validate(validator_address, current_time)
            return False
            
    def _create_new_block(self) -> Optional[Dict]:
        """Create a new block"""
        try:
            # Select transactions from mempool
            transactions = self._select_transactions_for_block()
            # Get current validator address
            if not self.wallet or not self.wallet.addresses:
            	return None
            
            # Get current validator
            validator_address = list(self.wallet.addresses.keys())[0]
            
            # Create block header
            previous_block = self.blockchain[-1]
            new_block = {
                'height': previous_block['height'] + 1,
                'previous_hash': previous_block['hash'],
                'timestamp': int(time.time()),
                'difficulty': self.current_difficulty,
                'validator': validator_address,
                'transactions': transactions,
                'version': 2,
                'nonce': 0
            }
            
            # Calculate merkle root
            new_block['merkle_root'] = self._calculate_merkle_root(transactions)
            
            # Sign the block (Proof-of-Stake)
            block_hash = self._calculate_block_hash(new_block)
            signature = self.wallet.sign_data(block_hash.encode())
            new_block['signature'] = signature
            new_block['hash'] = block_hash
            
            # Add block reward transaction
            reward_tx = self._create_block_reward_transaction(validator_address)
            new_block['transactions'].insert(0, reward_tx)
            
            return new_block
            
        except Exception as e:
            logger.error(f"Block creation failed: {e}")
            return None
    
    def _select_transactions_for_block(self) -> List[Dict]:
        """Select transactions for new block"""
        # Sort by fee rate (higher fees first)
        sorted_txs = sorted(self.mempool, key=lambda tx: tx.get('fee_rate', 0), reverse=True)
        
        selected_txs = []
        current_size = 0
        
        for tx in sorted_txs:
            tx_size = self._calculate_transaction_size(tx)
            if current_size + tx_size <= self.config['max_block_size']:
                if self._validate_transaction(tx):
                    selected_txs.append(tx)
                    current_size += tx_size
            
            if current_size >= self.config['max_block_size']:
                break
        
        return selected_txs
    
    def _create_block_reward_transaction(self, validator_address: str) -> Dict:
        """Create block reward transaction"""
        block_reward = self._get_block_reward()
        
        # Calculate foundation fee
        foundation_fee = int(block_reward * self.config['developer_fee_percent'])
        validator_reward = block_reward - foundation_fee
        
        # Create reward transaction
        reward_tx = Transaction(
            inputs=[],
            outputs=[
                {
                    'address': validator_address,
                    'amount': validator_reward,
                    'locktime': 0
                },
                {
                    'address': self.config['foundation_address'],
                    'amount': foundation_fee,
                    'locktime': 0
                }
            ],
            locktime=0
        )
        
        # Update supply tracking
        self.total_supply += block_reward
        self.circulating_supply += validator_reward
        self.foundation_funds += foundation_fee
        self.staking_rewards_distributed += validator_reward
        
        return reward_tx
    
    def _get_block_reward(self) -> int:
        """Calculate current block reward with halving"""
        height = len(self.blockchain)
        halvings = height // self.config['halving_interval']
        
        # Base reward divided by 2^halvings
        reward = self.config['block_reward'] >> halvings
        
        # Ensure reward doesn't go below minimum
        return max(reward, 1)
    
    def _add_block(self, block: Dict):
        """Add block to blockchain"""
        # Validate block
        if not self._validate_block(block):
            raise ValueError("Invalid block")
        
        # Add to blockchain
        self.blockchain.append(block)
        
        # Update UTXO set
        self._update_utxo_set(block)
        
        # Remove transactions from mempool
        self._remove_transactions_from_mempool(block['transactions'])
        
        # Adjust difficulty if needed
        self._adjust_difficulty()
        
        # Save to database
        self._save_blockchain()
        
        logger.info(f"New block added: #{block['height']} - {block['hash'][:16]}...")
    
    def _validate_block(self, block: Dict) -> bool:
        """Validate block"""
        try:
            # Check block structure
            required_fields = ['height', 'previous_hash', 'timestamp', 'difficulty', 
                             'validator', 'transactions', 'merkle_root', 'hash', 'signature']
            for field in required_fields:
                if field not in block:
                    return False
            
            # Check block hash
            calculated_hash = self._calculate_block_hash(block)
            if calculated_hash != block['hash']:
                return False
            
            # Check previous block
            if block['previous_hash'] != self.blockchain[-1]['hash']:
                return False
            
            # Check merkle root
            calculated_merkle = self._calculate_merkle_root(block['transactions'])
            if calculated_merkle != block['merkle_root']:
                return False
            
            # Check validator signature
            if not self._validate_block_signature(block):
                return False
            
            # Validate all transactions
            for tx in block['transactions']:
                if not self._validate_transaction(tx):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Block validation failed: {e}")
            return False
    
    def _validate_block_signature(self, block: Dict) -> bool:
        """Validate block signature"""
        try:
            # Get validator public key
            validator_address = block['validator']
            if self.wallet:
            	public_key = self.wallet.get_public_key_for_address(validator_address)
            	if not public_key:
            		return False
            	
            else:
            	# Fallback: get from consensus (if implemented)
            	public_key = self.consensus.get_validator_public_key(validator_address)
            	if not public_key:
            		return False
            
            
            # Verify signature
            block_data = self._get_block_signing_data(block)
            return self.wallet.verify_signature(block_data, block['signature'], public_key)
            
        except Exception as e:
            logger.error(f"Signature validation failed: {e}")
            return False
    
    def _get_block_signing_data(self, block: Dict) -> bytes:
        """Get data used for block signing"""
        # Exclude signature from signing data
        signing_block = block.copy()
        if 'signature' in signing_block:
            del signing_block['signature']
        
        return json.dumps(signing_block, sort_keys=True).encode()
    
    def _calculate_block_hash(self, block: Dict) -> str:
        """Calculate block hash"""
        block_data = self._get_block_signing_data(block)
        return hashlib.sha256(block_data).hexdigest()
    
    def _calculate_merkle_root(self, transactions: List[Dict]) -> str:
        """Calculate merkle root of transactions"""
        if not transactions:
            return '0' * 64
            
        # Convert Transaction objects to dicts if needed
        tx_dicts = []
        for tx in transactions:
        	if hasattr(tx, 'to_dict'):
        		tx_dicts.append(tx.to_dict())
        	else:
        		tx_dicts.append(tx)  
        
        tx_hashes = [self._calculate_transaction_hash(tx) for tx in tx_dicts]
        merkle_tree = MerkleTree(tx_hashes)
        return merkle_tree.get_root_hash()
    
    def _calculate_transaction_hash(self, transaction: Dict) -> str:
        """Calculate transaction hash"""
        if hasattr(transaction, 'to_dict'):
        	tx_data = transaction.to_dict()
        else:
        	tx_data = transaction
        	
        sorted_data = json.dumps(tx_data, sort_keys=True).encode()
        return hashlib.sha256(sorted_data).hexdigest()
    
    def _calculate_transaction_size(self, transaction: Dict) -> int:
        """Calculate transaction size in bytes"""
        return len(json.dumps(transaction).encode())
    
    def _update_utxo_set(self, block: Dict):
        """Update UTXO set with block transactions"""
        for tx_data in block['transactions']:
            # Convert dict to Transaction object if needed
            if isinstance(tx_data, dict):
            	transaction = Transaction.from_dict(tx_data)
            	
            else:
            	transaction = tx_data
            
            self.utxo_set.process_transaction(transaction)
    
    def _remove_transactions_from_mempool(self, transactions: List[Dict]):
        """Remove transactions from mempool"""
        tx_hashes = [self._calculate_transaction_hash(tx) for tx in transactions]
        self.mempool = [tx for tx in self.mempool 
                       if self._calculate_transaction_hash(tx) not in tx_hashes]
    
    def _adjust_difficulty(self):
        """Adjust mining difficulty"""
        if len(self.blockchain) % self.config['difficulty_adjustment_blocks'] == 0:
            self._recalculate_difficulty()
    
    def _recalculate_difficulty(self):
        """Recalculate current difficulty"""
        # Get blocks from previous difficulty period
        start_height = max(0, len(self.blockchain) - self.config['difficulty_adjustment_blocks'])
        adjustment_blocks = self.blockchain[start_height:]
        
        if len(adjustment_blocks) < 2:
            return
        
        # Calculate actual time taken
        actual_time = adjustment_blocks[-1]['timestamp'] - adjustment_blocks[0]['timestamp']
        target_time = self.config['block_time_target'] * len(adjustment_blocks)
        
        # Adjust difficulty
        ratio = actual_time / target_time
        if ratio < 0.5:
            ratio = 0.5
        elif ratio > 2.0:
            ratio = 2.0
        
        new_difficulty = self.current_difficulty * ratio
        self.current_difficulty = max(1, int(new_difficulty))
        
        logger.info(f"Difficulty adjusted: {self.current_difficulty}")
    
    def _validate_transaction(self, transaction: Dict) -> bool:
        """Validate transaction"""
        try:
            tx = Transaction.from_dict(transaction)
            
            # Check basic structure
            if not tx.inputs or not tx.outputs:
                return False
            
            # Check transaction size
            if self._calculate_transaction_size(transaction) > self.config['max_transaction_size']:
                return False
            
            # Check fees
            total_input = 0
            total_output = 0
            
            # Validate inputs
            for tx_input in tx.inputs:
                # Check if UTXO exists and is unspent
                utxo = self.utxo_set.get_utxo(tx_input['tx_hash'], tx_input['output_index'])
                if not utxo or utxo.spent:
                    return False
                
                # Check input signature
                if not self._validate_input_signature(tx_input, utxo):
                    return False
                
                total_input += utxo.amount
            
            # Calculate outputs
            for output in tx.outputs:
                total_output += output['amount']
            
            # Check if outputs don't exceed inputs
            if total_output > total_input:
                return False
            
            # Check minimum fee
            fee = total_input - total_output
            if fee < self.config['min_transaction_fee']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Transaction validation failed: {e}")
            return False
    
    def _validate_input_signature(self, tx_input: Dict, utxo: UTXO) -> bool:
        """Validate transaction input signature"""
        # This would verify the cryptographic signature
        # For now, we'll simulate validation
        return True
        
    def get_validator_info(self, validator_address: Optional[str] = None) -> Dict:
    	start_time = time.time()
    	metrics = {
    	    'method_calls': {},
    	    'timings': {},
    	    'errors': []
    	}
    	try:
    		# Input validation and sanitization
    		if validator_address:
    			if not self._validate_address_format(validator_address):
    				return self._create_error_response(
    				    'invalid_address_format',
    				    f"Invalid validator address format: {validator_address}",
    				    metrics
    				)
    			validator_address = validator_address.strip().lower()
    		# Address resolution with fallbacks
    		resolved_address = self._resolve_validator_address(validator_address)
    		if not resolved_address:
    			return self._create_error_response(
    			    'address_resolution_failed',
    			    'No validator address could be resolved',
    			    metrics
    			)
    		validator_address = resolved_address
    		
    		# Check if validator exists in consensus
    		if not self.consensus.validator_exists(validator_address):
    			return self._create_error_response(
    			    'validator_not_found',
    			    f'Validator {validator_address} not found in consensus',
    			    metrics,
    			    {'validator_address': validator_address}
    			)
    		# Parallel data collection for performance
    		with ThreadPoolExecutor(max_workers=4) as executor:
    			# Submit all data collection tasks
    			basic_info_future = executor.submit(
    			self._get_basic_validator_info, validator_address
    			)
    			staking_info_future = executor.submit(
    			self._get_staking_info, validator_address
    			)
    			performance_future = executor.submit(
    			self._calculate_validator_performance, validator_address
    			)
    			economics_future = executor.submit(
    			self._get_validator_economics, validator_address
    			)
    			# Wait for completion with timeout
    			basic_info = self._get_with_timeout(basic_info_future, 5, 'basic_info')
    			staking_info = self._get_with_timeout(staking_info_future, 5, 'staking_info')
    			performance = self._get_with_timeout(performance_future, 10, 'performance')
    			economics = self._get_with_timeout(economics_future, 8, 'economics')
    		# Get additional info sequentially
    		schedule = self._get_validation_schedule(validator_address)
    		network_status = self._get_network_validator_status()
    		slashing_risk = self._assess_slashing_risk(validator_address)
    		# Compile comprehensive response
    		response = {
    		    'validator_address': validator_address,
    		    'basic_info': basic_info,
    		    'staking_info': staking_info,
    		    'performance_metrics': performance,
    		    'economics': economics,
    		    'validation_schedule': schedule,
    		    'network_status': network_status,
    		    'slashing_risk': slashing_risk,
    		    'system_info': {
    		        'wallet_connected': self.wallet is not None,
    		        'network_connected': self.network is not None and self.network.is_connected(),
    		        'current_height': len(self.blockchain) - 1,
    		        'timestamp': int(time.time()),
    		        'response_time_ms': int((time.time() - start_time) * 1000)
    		    },
    		    'metadata': {
    		        'version': '2.0',
    		        'format': 'standardized',
    		        'cache_status': 'live',
    		        'data_freshness': self._get_data_freshness()
    		    }
    		}
    		# Add signatures for data integrity
    		response['signature'] = self._sign_validator_info(response)
    		
    		# Update metrics
    		metrics['timings']['total'] = time.time() - start_time
    		response['_metrics'] = metrics
    		
    		# Cache the response
    		self._cache_validator_info(validator_address, response)
    		
    		return response
    		
    	except Exception as e:
    	    logger.error(f"Critical error in get_validator_info: {e}", exc_info=True)
    	    metrics['errors'].append({
    	        'type': 'critical',
    	        'message': str(e),
    	        'timestamp': time.time()
    	    })
    	    return self._create_error_response(
    	        'internal_error',
    	        f'Internal server error: {str(e)}',
    	        metrics,
    	        {'validator_address': validator_address}
    	    )		
    			
    def _resolve_validator_address(self, validator_address: Optional[str]) -> Optional[str]:
    	"""Resolve validator address with multiple fallback strategies."""
            	
    	try:
    		# If address provided, validate and return
    		if validator_address:
    			if self._validate_address_format(validator_address):
    				return validator_address.lower()
    			return None
    			
    		# Fallback 1: Use wallet's primary address
    		if self.wallet and self.wallet.addresses:
    			primary_address = list(self.wallet.addresses.keys())[0]
    			if self._validate_address_format(primary_address):
    				return primary_address.lower()
    				
    		# Fallback 2: Check if we're a validator in consensus
    		if hasattr(self.consensus, 'get_local_validator_address'):
    	         local_address = self.consensus.get_local_validator_address()
    	         if local_address and self._validate_address_format(local_address):
    	         	
    	         	return local_address.lower()
    	         	# Fallback 3: Check database for recently used validator
    	         	cached_validator = self.database.get('last_used_validator')
    	         	if cached_validator and self._validate_address_format(cached_validator):
    	         		return cached_validator.lower()
    	         	return None
    	         	
    	except Exception as e:
    	    logger.warning(f"Address resolution failed: {e}")
    	    return None   
    	 
                 
        	
    def _get_basic_validator_info(self, validator_address: str) -> Dict:
    	try:
    		info = self.consensus.get_validator_info(validator_address)
    		
    		# Enhanced validation of returned data
    		required_fields = ['registered', 'active', 'registration_height']
    		for field in required_fields:
    			if field not in info:
    				raise ValueError(f"Missing required field: {field}")
    				
    		# Additional derived information
    		current_height = len(self.blockchain) - 1
    		age_blocks = current_height - info.get('registration_height', current_height)
    		return {
    		    'is_registered': bool(info.get('registered', False)),
    		    'is_active': bool(info.get('active', False)),
    		    'registration_height': info.get('registration_height'),
    		    'last_active_height': info.get('last_active_height'),
    		    'validator_age_blocks': max(0, age_blocks),
    		    'validator_age_days': self._blocks_to_days(age_blocks),
    		    'public_key': info.get('public_key'),
    		    'commission_rate': info.get('commission_rate', 0.0),
    		    'max_commission_rate': info.get('max_commission_rate', 0.0),
    		    'commission_change_rate': info.get('commission_change_rate', 0.0),
    		    'identity': info.get('identity', {}),
    		    'description': info.get('description', {}),
    		    'status': self._determine_validator_status(info)
    		}
    	except Exception as e:
    		logger.error(f"Failed to get basic validator info: {e}")
    		return {
    		    'error': f'basic_info_failure: {str(e)}',
    		    'is_registered': False,
    		    'is_active': False,
    		    'status': 'unknown'
    		}    	         	

    def _get_staking_info(self, validator_address: str) -> Dict:
    	try:
    	    staked_amount = self.consensus.get_stake_amount(validator_address)
    	    total_staked = self.consensus.get_total_stake()
    	    delegated_stake = self.consensus.get_delegated_stake(validator_address)
    	    
    	    # Calculate percentages with safety checks 
    	    stake_percentage = (staked_amount / total_staked * 100) if total_staked > 0 else 0
    	    voting_power = self.consensus.get_voting_power(validator_address) if hasattr(
            self.consensus, 'get_voting_power') else staked_amount
    	    return {
    	        'self_stake': staked_amount,
    	        'delegated_stake': delegated_stake,
    	        'total_stake': staked_amount + delegated_stake,
    	        'total_network_stake': total_staked,
    	        'stake_percentage': round(stake_percentage, 4),
    	        'voting_power': voting_power,
    	        'voting_power_percentage': (voting_power / total_staked * 100) if total_staked > 0 else 0,
    	        'min_self_stake_requirement': self.config['stake_minimum'],
    	        'meets_min_stake': staked_amount >= self.config['stake_minimum'],
    	        'stake_rank': self.consensus.get_validator_rank(validator_address) if hasattr(
                self.consensus, 'get_validator_rank') else None,
                'delegators_count': self.consensus.get_delegators_count(validator_address) if hasattr(
                self.consensus, 'get_delegators_count') else 0
    	    }
    	except Exception as e:
    	    logger.error(f"Failed to get staking info: {e}")
    	    return {
    	        'error': f'staking_info_failure: {str(e)}',
    	        'self_stake': 0,
    	        'total_network_stake': 0,
    	        'stake_percentage': 0
    	    }	
    	    
    def _calculate_validator_performance(self, validator_address: str) -> Dict:
        """Calculate comprehensive performance metrics with caching."""
        cache_key = f"validator_perf_{validator_address}"
        cached = self._get_cached_performance(cache_key)
        if cached and not self._is_performance_data_stale(cached):
            return cached
        try:
            # Count blocks validated with efficient scanning
            blocks_validated = 0
            recent_blocks = 0
            total_blocks = max(1, len(self.blockchain) - 1)
            
            # Use efficient block scanning for large blockchains
            if len(self.blockchain) > 1000:
            	# For large chains, sample recent blocks and extrapolate
            	sample_size = min(1000, total_blocks)
            	sample_blocks = self.blockchain[-sample_size:]
            	sample_validated = sum(1 for block in sample_blocks if block.get('validator') == validator_address)
            	blocks_validated = int(sample_validated * (total_blocks / sample_size))
            else:
                # For small chains, scan all blocks
                blocks_validated = sum(1 for block in self.blockchain[1:] if block.get('validator') == validator_address)
            # Get detailed performance metrics
            uptime = self.consensus.get_validator_uptime(validator_address)
            block_times = self._get_validator_block_times(validator_address)
            recent_activity = self._get_recent_validator_activity(validator_address)
            performance_data = {
                'blocks_validated': blocks_validated,
                'validation_success_rate': round(
                    (blocks_validated / total_blocks * 100) if total_blocks > 0 else 0, 2
                ),
                'uptime_percentage': round(uptime or 0, 2),
                'average_block_time': round(
                    sum(block_times) / len(block_times) if block_times else 0, 3
                ),
                'block_time_std_dev': round(
                    statistics.stdev(block_times) if len(block_times) > 1 else 0, 3
                ),
                'missed_blocks': self.consensus.get_missed_blocks(validator_address),
                'last_missed_block': self.consensus.get_last_missed_block(validator_address),
                'consecutive_successes': self.consensus.get_consecutive_successes(validator_address),
                'recent_activity': recent_activity,
                'performance_score': self._calculate_performance_score(
                    blocks_validated, uptime, block_times
                ),
                'reliability_rating': self._get_reliability_rating(uptime or 0),
                'efficiency_rating': self._get_efficiency_rating(block_times)
            
            }
            # Cache the performance data
            self._cache_performance_data(cache_key, performance_data)
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to calculate performance: {e}")
            return {
                'error': f'performance_calculation_failure: {str(e)}',
                'blocks_validated': 0,
                'validation_success_rate': 0,
                'uptime_percentage': 0
            }                                              
                                 
    def _get_validator_economics(self, validator_address: str) -> Dict:
    	"""Get comprehensive economic information with forecasting."""
    	try:
    		staked_amount = self.consensus.get_stake_amount(validator_address)
    		total_rewards = self._calculate_validator_rewards(validator_address)
    		pending_rewards = self._calculate_pending_rewards(validator_address)
    		total_stake = self.consensus.get_total_stake()
    		
    		# Calculate advanced metrics
    		apr = self._calculate_apr(validator_address, total_rewards)
    		roi = self._calculate_roi(validator_address, total_rewards, staked_amount)
    		annual_earnings = self._estimate_annual_earnings(validator_address)
    		return {
    		    'staked_amount': staked_amount,
    		    'total_rewards_earned': total_rewards,
    		    'pending_rewards': pending_rewards,
    		    'unclaimed_rewards': self.consensus.get_unclaimed_rewards(validator_address),
    		    'reward_history': self._get_reward_history(validator_address),
    		    'apr': round(apr, 4),
    		    'apy': round(self._calculate_apy(apr), 4),
    		    'roi': round(roi, 4),
    		    'annual_earnings_estimate': annual_earnings,
    		    'fee_structure': self._get_fee_structure(),
    		    'commission_earnings': self._calculate_commission_earnings(validator_address),
    		    
    		    'delegator_rewards': self._calculate_delegator_rewards(validator_address),
    		    'economic_viability': self._assess_economic_viability(
                staked_amount, apr, total_stake
            ),
    		    'break_even_estimation': self._calculate_break_even(validator_address)
        
    		    
    		}
    	except Exception as e:
    		logger.error(f"Failed to get economics info: {e}")
    		return {
    		    'error': f'economics_calculation_failure: {str(e)}',
    		    'staked_amount': 0,
    		    'total_rewards_earned': 0,
    		    'apr': 0
    		}  
    
    def _assess_slashing_risk(self, validator_address: str) -> Dict:    	
    	                      
        try:
        	# Get comprehensive risk factors
        	risk_factors = self._collect_risk_factors(validator_address)
        	
        	# Calculate risk score using weighted factors
        	risk_score = self._calculate_risk_score(risk_factors)
        	risk_level = self._determine_risk_level(risk_score)
        	
        	# Get mitigation recommendations
        	recommendations = self._get_risk_recommendations(risk_level, risk_factors)
        	
        	# Calculate financial impact
        	financial_impact = self._calculate_slashing_impact(validator_address, risk_score)
        	return {
        	    'risk_level': risk_level,
        	    'risk_score': round(risk_score, 4),
        	    'risk_factors': risk_factors,
        	    'financial_impact': financial_impact,
        	    'recommendations': recommendations,
        	    'monitoring_required': risk_level in ['high', 'critical'],
        	    'last_risk_assessment': int(time.time()),
        	    'risk_trend': self._get_risk_trend(validator_address),
        	    'insurance_coverage': self._get_insurance_coverage(validator_address)
        	}
        except Exception as e:
            logger.error(f"Failed to assess slashing risk: {e}")
            return {
                
                'risk_level': 'unknown',
                'error': f'risk_assessment_failure: {str(e)}'
            }	        	                   

    def _validate_address_format(self, address: str) -> bool:
    	"""Validate RAYONIX address format with comprehensive checks"""
    	if not address or not isinstance(address, str):
    		return False
    		
    	# Check basic format requirements
    	if not address.startswith('ryx1') or len(address) < 42 or len(address) > 90:
    		return False
    		
    		# Bech32 validation with detailed error checking
    		try:
    			hrp, data = bech32.bech32_decode(address)
    			if hrp != 'ryx' or data is None:
    				return False
    				
    			# Additional format validation
    			if not bech32.bech32_verify_checksum(hrp, data):
    				return False
    				
    			# Check for valid witness version and program length
    			decoded = bech32.decode(hrp, address)
    			if decoded is None:
    				return False
    				
    			witness_version, witness_program = decoded
    			if witness_version not in [0, 1]:
    				return False
    				
    			# Validate program length based on witness version
    			if witness_version == 0:
    				if len(witness_program) not in [20, 32]:
    					return False
    			elif witness_version == 1:
    				if len(witness_program) != 32:
    					return False
    					
    			return True
    			
    		except (ValueError, TypeError, Exception) as e:
    			logger.debug(f"Address validation failed for {address}: {e}")
    			return False
    			
    def _create_error_response(self, error_code: str, message: str, metrics: Dict, extra: Dict = None) -> Dict:
    	"""Create comprehensive standardized error response"""    			        		        	                 		        	         
    def _sync_loop(self):
        """Blockchain synchronization loop"""
        while True:
            try:
                if self.network and self.network.is_connected():
                    self._synchronize_with_network()
                
                time.sleep(30)  # Sync every 30 seconds
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
                time.sleep(60)
    
    def _synchronize_with_network(self):
        """Synchronize with network peers"""
        # Get best block from peers
        best_block = self._get_best_block_from_peers()
        
        if best_block and best_block['height'] > len(self.blockchain) - 1:
            # We need to sync
            self._download_missing_blocks(best_block['height'])
    
    def _get_best_block_from_peers(self) -> Optional[Dict]:
        """Get best block information from network peers"""
        # This would query multiple peers and return consensus best block
        # For now, return None (simulation)
        return None
    
    def _download_missing_blocks(self, target_height: int):
        """Download missing blocks from network"""
        current_height = len(self.blockchain) - 1
        
        while current_height < target_height:
            try:
                next_height = current_height + 1
                block = self._request_block_from_peers(next_height)
                
                if block and self._validate_block(block):
                    self._add_block(block)
                    current_height += 1
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Failed to download block {next_height}: {e}")
                break
    
    def _request_block_from_peers(self, height: int) -> Optional[Dict]:
        """Request block from network peers"""
        # This would send network requests for specific block
        # For now, return None (simulation)
        return None
    
    def _mempool_loop(self):
        """Mempool management loop"""
        while True:
            try:
                # Clean old transactions
                self._clean_mempool()
                
                # Validate all transactions
                self._validate_mempool()
                
                # Broadcast transactions
                self._broadcast_mempool()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Mempool error: {e}")
                time.sleep(30)
    
    def _clean_mempool(self):
        """Clean old transactions from mempool"""
        current_time = time.time()
        self.mempool = [tx for tx in self.mempool 
                       if current_time - tx.get('timestamp', 0) < 3600]  # 1 hour
    
    def _validate_mempool(self):
        """Validate all transactions in mempool"""
        valid_transactions = []
        
        for tx in self.mempool:
            if self._validate_transaction(tx):
                valid_transactions.append(tx)
        
        self.mempool = valid_transactions
    
    def _broadcast_mempool(self):
        """Broadcast mempool transactions to network"""
        if self.network and self.mempool:
            for tx in self.mempool[:10]:  # Broadcast first 10
                self._broadcast_transaction(tx)
    
    def _broadcast_block(self, block: Dict):
        """Broadcast block to network"""
        if self.network:
            self.network.broadcast_message('block', block)
    
    def _broadcast_transaction(self, transaction: Dict):
        """Broadcast transaction to network"""
        if self.network:
            self.network.broadcast_message('transaction', transaction)
    
    def _handle_block_message(self, message: Dict):
        """Handle incoming block message"""
        try:
            block = message.payload
            if self._validate_block(block):
                self._add_block(block)
        except Exception as e:
            logger.error(f"Block message handling failed: {e}")
    
    def _handle_transaction_message(self, message: Dict):
        """Handle incoming transaction message"""
        try:
            transaction = message.payload
            if self._validate_transaction(transaction):
                self._add_to_mempool(transaction)
        except Exception as e:
            logger.error(f"Transaction message handling failed: {e}")
    
    def _handle_consensus_message(self, message: Dict):
        """Handle consensus message"""
        try:
            # Process consensus messages (votes, proposals, etc.)
            # This would be implemented based on specific consensus protocol
            pass
        except Exception as e:
            logger.error(f"Consensus message handling failed: {e}")
    
    def _add_to_mempool(self, transaction: Dict):
        """Add transaction to mempool"""
        # Check if already in mempool
        tx_hash = self._calculate_transaction_hash(transaction)
        existing_tx = next((tx for tx in self.mempool 
                          if self._calculate_transaction_hash(tx) == tx_hash), None)
        
        if not existing_tx:
            transaction['timestamp'] = time.time()
            transaction['fee_rate'] = self._calculate_fee_rate(transaction)
            self.mempool.append(transaction)
    
    def _calculate_fee_rate(self, transaction: Dict) -> float:
        """Calculate fee rate (fee per byte)"""
        tx_size = self._calculate_transaction_size(transaction)
        
        # Calculate total fee
        total_input = sum(inp.get('amount', 0) for inp in transaction.get('inputs', []))
        total_output = sum(out.get('amount', 0) for out in transaction.get('outputs', []))
        fee = total_input - total_output
        
        return fee / tx_size if tx_size > 0 else 0
    
    def _calculate_supply(self):
        """Calculate current supply from blockchain"""
        self.total_supply = 0
        self.circulating_supply = 0
        self.staking_rewards_distributed = 0
        self.foundation_funds = 0
        
        for block in self.blockchain:
            for tx in block['transactions']:
                # Skip coinbase transactions for input calculation
                if not tx.get('inputs'):
                    # This is a coinbase transaction
                    total_output = sum(out.get('amount', 0) for out in tx.get('outputs', []))
                    self.total_supply += total_output
                    
                    # Track foundation funds
                    for output in tx.get('outputs', []):
                        if output.get('address') == self.config['foundation_address']:
                            self.foundation_funds += output.get('amount', 0)
                        else:
                            self.circulating_supply += output.get('amount', 0)
                            self.staking_rewards_distributed += output.get('amount', 0)
    
    def create_transaction(self, from_address: str, to_address: str, amount: int, 
                          fee: Optional[int] = None, memo: Optional[str] = None) -> Optional[Dict]:
        """Create and sign a transaction"""
        try:
            # Get UTXOs for sender
            utxos = self.utxo_set.get_utxos_for_address(from_address)
            if not utxos:
                raise ValueError("No spendable funds")
            
            # Calculate total available
            total_available = sum(utxo.amount for utxo in utxos)
            
            # Set default fee if not provided
            if fee is None:
                fee = self.config['min_transaction_fee']
            
            # Check if sufficient funds
            if total_available < amount + fee:
                raise ValueError("Insufficient funds")
            
            # Select UTXOs to spend
            selected_utxos = []
            selected_amount = 0
            
            for utxo in sorted(utxos, key=lambda x: x.amount, reverse=True):
                if selected_amount >= amount + fee:
                    break
                selected_utxos.append(utxo)
                selected_amount += utxo.amount
            
            # Create transaction inputs
            inputs = []
            for utxo in selected_utxos:
                inputs.append({
                    'tx_hash': utxo.tx_hash,
                    'output_index': utxo.output_index,
                    'address': from_address,
                    'amount': utxo.amount
                })
            
            # Create transaction outputs
            outputs = [
                {
                    'address': to_address,
                    'amount': amount,
                    'locktime': 0
                }
            ]
            
            # Add change output if needed
            change_amount = selected_amount - amount - fee
            if change_amount > 0:
                # Generate a new change address from the wallet
                if self.wallet:
             
                	change_address_info = self.wallet.derive_address(len(self.wallet.addresses), True)
                	change_address = change_address_info.address
            else:
                # Fallback: use from_address for change
                change_address = from_address
                
                outputs.append({
                    'address': change_address,
                    'amount': change_amount,
                    'locktime': 0
                })
            
            # Create transaction
            transaction = Transaction(
                inputs=inputs,
                outputs=outputs,
                locktime=0
            )
            
            # Sign transaction - use the wallet's sign_transaction method
            transaction_dict = transaction.to_dict()
            if self.wallet:
            	signature = self.wallet.sign_transaction(transaction_dict)
            	transaction_dict['signature'] = signature
            
            # Add to mempool
            self._add_to_mempool(transaction_dict)
            
            return transaction_dict
            
        except Exception as e:
            logger.error(f"Transaction creation failed: {e}")
            return None
    
    def get_balance(self, address: str) -> int:
        """Get balance for address"""
        utxos = self.utxo_set.get_utxos_for_address(address)
        return sum(utxo.amount for utxo in utxos)
    
    def get_blockchain_info(self) -> Dict:
        """Get blockchain information"""
        return {
            'height': len(self.blockchain) - 1,
            'difficulty': self.current_difficulty,
            'total_supply': self.total_supply,
            'circulating_supply': self.circulating_supply,
            'block_reward': self._get_block_reward(),
            'mempool_size': len(self.mempool),
            'foundation_funds': self.foundation_funds,
            'staking_rewards': self.staking_rewards_distributed,
            'network': self.network_type,
            'version': 1
        }
    
    def get_block(self, height: int) -> Optional[Dict]:
        """Get block by height"""
        if 0 <= height < len(self.blockchain):
            return self.blockchain[height]
        return None
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction by hash"""
        # Check mempool first
        for tx in self.mempool:
            if self._calculate_transaction_hash(tx) == tx_hash:
                return tx
        
        # Check blockchain
        for block in self.blockchain:
            for tx in block['transactions']:
                if self._calculate_transaction_hash(tx) == tx_hash:
                    return tx
        
        return None
    
    def start_mining(self):
        """Start mining/staking"""
        if not self.mining_thread.is_alive():
            self.mining_thread = threading.Thread(target=self._mining_loop, daemon=True)
            self.mining_thread.start()
            logger.info("Mining started")
    
    def stop_mining(self):
        """Stop mining/staking"""
        # Mining loop checks a flag, so we just need to set it
        # This would be implemented with proper threading control
        logger.info("Mining stopped")
    
    def connect_to_network(self):
        """Connect to P2P network"""
        if self.network:
            self.network.start()
            logger.info("Network connection started")
    
    def disconnect_from_network(self):
        """Disconnect from P2P network"""
        if self.network:
            self.network.stop()
            logger.info("Network connection stopped")
    
    def deploy_contract(self, contract_code: str, initial_balance: int = 0) -> Optional[str]:
        """Deploy smart contract"""
        if not self.wallet:
            raise ValueError("Wallet not available")
        
        return self.contract_manager.deploy_contract(
            self.wallet.get_address(),
            contract_code,
            initial_balance
        )
    
    def call_contract(self, contract_address: str, function_name: str, 
                     args: List[Any], value: int = 0) -> Any:
        """Call smart contract function"""
        if not self.wallet:
            raise ValueError("Wallet not available")
        
        return self.contract_manager.execute_contract(
            contract_address,
            function_name,
            args,
            self.wallet.get_address(),
            value
        )
    
    def register_validator(self, stake_amount: int) -> bool:
        """Register as validator"""
        if not self.wallet or not self.wallet.addresses:
            raise ValueError("Wallet not available or no addresses")
        validator_address = list(self.wallet.addresses.keys())[0]
        
        # Check minimum stake
        if stake_amount < self.config['stake_minimum']:
            raise ValueError(f"Minimum stake is {self.config['stake_minimum']} RXY")
        
        # Check balance
        balance = self.get_balance(validator_address)
        if balance < stake_amount:
            raise ValueError("Insufficient balance")
        
        # Create staking transaction
        staking_tx = self.create_transaction(
        validator_address,
        self.config['foundation_address'],  # Staking contract address
        stake_amount,
        fee=self.config['min_transaction_fee'],
        memo="Validator registration"
    )
        
        if staking_tx:
            # Get public key from wallet
            public_key = self.wallet.get_public_key_for_address(validator_address)
            if not public_key:
            	raise ValueError("Could not retrieve public key")
            	
            # Register with consensus	
            return self.consensus.register_validator(
            validator_address,
            public_key.hex(),  # Convert bytes to hex string
            stake_amount
        )
        
        return False

    def close(self):
        """Cleanup resources"""
        if hasattr(self, 'database'):
            self.database.close()
        if hasattr(self, 'network') and self.network is not None:
            self.network.stop()

    def __del__(self):
        """Destructor"""
        self.close()

# Utility functions
    def create_rayonix_network(network_type: str = "mainnet") -> RayonixCoin:
        """Create RAYONIX network instance"""
        return RayonixCoin(network_type)

    def generate_genesis_block(config: Dict) -> Dict:
        """Generate genesis block with custom configuration"""
        # This would create a custom genesis block for private networks
        pass

    def validate_rayonix_address(address: str) -> bool:
        """Validate RAYONIX address"""
        # RAYONIX uses Bech32 addresses starting with 'ryx'
        if not address or not isinstance(address, str):
            return False
        
        # Check length and prefix
        if not address.startswith('ryx1') or len(address) != 42:
            return False
        
        # Bech32 validation
        try:
            hrp, data = bech32.decode(address)
            return hrp == 'ryx' and data is not None
        except:
            return False
        
    def _get_with_timeout(self, future, timeout: int, operation: str) -> Any:
        """Get future result with timeout and error handling."""
        try:
        	return future.result(timeout=timeout)
        except TimeoutError:
        	logger.warning(f"Timeout in {operation} operation")
        	
        	return {'error': f'timeout_in_{operation}', 'timed_out': True}
        	
        except Exception as e:
        	 logger.error(f"Error in {operation} operation: {e}")
        	 
        	 return {'error': f'error_in_{operation}: {str(e)}'}
    
        
    def _create_error_response(self, error_code: str, message: str, metrics: Dict, extra: Dict = None) -> Dict:
    	"""Create standardized error response."""
    	error_data = {
    	    'code': error_code,
    	    'message': message,
    	    'timestamp': int(time.time()),
    	    'request_id': hashlib.sha256(f"{time.time()}{error_code}".encode()).hexdigest()[:16]
    	}
    	if extra:
    	    error_data.update(extra)
    	response = {
    	    'error': error_data,
    	    'system_info': {
    	        'current_height': len(self.blockchain) - 1,
    	        'blockchain_hash': self.blockchain[-1]['hash'] if self.blockchain else '0' * 64,
    	        'network_id': self.config['network_id'],
    	        'node_version': '1.0.0',
    	        'timestamp': int(time.time()),
    	        'uptime_seconds': int(time.time() - self.start_time) if hasattr(self, 'start_time') else 0
    	    },
    	    '_metrics': metrics,
    	    'metadata': {
    	        'version': '2.0',
    	        'format': 'standardized_error',
    	        'suggested_actions': self._get_error_suggestions(error_code)
    	    }
    	}
    	return response
    	    
    def _get_error_suggestions(self, error_code: str) -> List[str]:
    	"""Get suggested actions for different error codes"""
    	suggestions = {
    	    'invalid_address_format': [
    	        "Check the address format starts with 'ryx1'",
    	        "Verify the address length is between 42-90 characters",
    	        "Ensure the address uses valid Bech32 encoding"
    	    ],
    	    'address_resolution_failed': [
    	        "Provide a specific validator address",
    	        "Ensure your wallet is properly initialized",
    	        "Check if the node has validator registration"
    	    ],
    	    'validator_not_found': [
    	        "Verify the validator address is correct",
    	        "Check if the validator is registered in consensus",
    	        "Ensure the validator has active stake"
    	    ],
    	    'internal_error': [
    	        "Check node logs for detailed error information",
    	        "Restart the node if the issue persists",
    	        "Verify database integrity"
    	    ]
    	}
    	return suggestions.get(error_code, ["Check system logs for details"])   

    def _get_with_timeout(self, future, timeout: int, operation: str) -> Any:
    	"""Robust future result retrieval with comprehensive timeout handling"""
    	try:
    		start_time = time.time()
    		result = future.result(timeout=timeout)
    		elapsed = time.time() - start_time
    		
    		if elapsed > timeout * 0.8:
    			logger.warning(f"Operation {operation} completed in {elapsed:.3f}s (close to timeout {timeout}s)")
    			
    		return result
    		
    	except TimeoutError:
    		logger.warning(f"Timeout in {operation} operation after {timeout}s")
    		return {
    		    'error': f'timeout_in_{operation}',
    		    'timed_out': True,
    		    'timeout_seconds': timeout,
    		    'suggestion': 'Consider increasing timeout or optimizing the operation'
    		}
    	except Exception as e:
    		logger.error(f"Error in {operation} operation: {e}", exc_info=True)
    		return {
    		    'error': f'error_in_{operation}',
    		    'exception_type': type(e).__name__,
    		    'exception_message': str(e),
    		    'timestamp': int(time.time())
    		}    
    		
    def _blocks_to_days(self, blocks: int) -> float:
    	"""Convert blocks to days with precise calculation"""
    	block_time = self.config['block_time_target']
    	return (blocks * block_time) / 86400.0 
    	# precise seconds per day    			 	        	 	    	 	        	 	    
    def _determine_validator_status(self, info: Dict) -> str:
    	"""Comprehensive validator status determination"""
    	if not info.get('registered', False):
    		return 'unregistered'
    		
    	if not info.get('active', False):
    		# Check specific reasons for inactivity
    		if info.get('jailed', False):
    			return 'jailed'
    		elif info.get('tombstoned', False):
    			return 'tombstoned'
    		elif info.get('unbonding', False):
    			return 'unbonding'
    		else:
    			return 'inactive'
    	# Check additional active status conditions
    	if info.get('missed_too_many_blocks', False):
    		return 'warning'
    	if info.get('low_commission', False):
    		return 'competitive'
    	if info.get('top_performer', False):
    		return 'elite'
    	return 'active'  
    	
    def _calculate_performance_score(self, blocks_validated: int, uptime: float, block_times: List[float]) -> float:
    	"""Calculate comprehensive performance score with multiple factors"""
    	if not block_times or blocks_validated == 0:
    		return 0.0
    	# Weighted scoring components
    	weights = {
    	    'uptime': 0.4,
    	    'consistency': 0.3,
    	    'efficiency': 0.2,
    	    'reliability': 0.1
    	}
    	
    	# Uptime component (0-1)
    	uptime_score = uptime
    	
    	# Consistency component (block time standard deviation)
    	avg_block_time = statistics.mean(block_times)
    	std_dev = statistics.stdev(block_times) if len(block_times) > 1 else 0
    	target_time = self.config['block_time_target']
    	consistency_score = max(0, 1 - (std_dev / target_time))
    	
    	# Efficiency component (how close to target block time)
    	efficiency_ratio = avg_block_time / target_time
    	efficiency_score = 1.0 / efficiency_ratio if efficiency_ratio > 1 else efficiency_ratio
    	
    	# Reliability component (consecutive successful blocks)
    	max_consecutive = self.consensus.get_consecutive_successes(self.validator_address)
    	reliability_score = min(1.0, max_consecutive / 100.0)
    	
    	# Calculate weighted total score
    	total_score = (
    	    weights['uptime'] * uptime_score +
    	    weights['consistency'] * consistency_score +
    	    weights['efficiency'] * efficiency_score +
    	    weights['reliability'] * reliability_score
    	)
    	return round(total_score, 4)

    def _get_reliability_rating(self, uptime: float) -> str:
    	"""Get detailed reliability rating with tiered thresholds"""
    	rating_map = [
    	    (0.999, 'platinum', 1.0),
    	    (0.995, 'gold', 0.9),
    	    (0.99, 'silver', 0.8),
    	    (0.98, 'bronze', 0.7),
    	    (0.95, 'standard', 0.6),
    	    (0.90, 'basic', 0.5),
    	    (0.80, 'unreliable', 0.3),
    	    (0.00, 'poor', 0.0)
    	]
    	
    	for threshold, rating, _ in rating_map:
    		if uptime >= threshold:
    			return rating 
    	return 'poor'    
  
    def _get_efficiency_rating(self, block_times: List[float]) -> str:
    	"""Get efficiency rating based on block time performance"""
    	if not block_times:
    		return 'unknown'
    		
    	avg_time = statistics.mean(block_times)
    	target = self.config['block_time_target']
    	ratio = avg_time / target
    	
    	efficiency_map = [
    	    (0.80, 'exceptional'),
    	    (0.90, 'excellent'),
    	    (0.95, 'very_good'),
    	    (1.00, 'good'),
    	    (1.05, 'average'),
    	    (1.10, 'below_average'),
    	    (1.20, 'poor'),
    	    (float('inf'), 'inefficient')
    	]
    	for threshold, rating, _ in rating_map:
    		if uptime >= threshold:
    			return rating 
    	return 'inefficient'
            
    def _calculate_apy(self, apr: float) -> float:
    	"""Calculate APY from APR with compound interest"""
    	compounding_periods = 365
    	apy = ((1 + (apr / compounding_periods)) ** compounding_periods) - 1
    	return round(apy, 6)
    	
    def _calculate_commission_earnings(self, validator_address: str) -> int:
    	"""Calculate total commission earnings for validator"""
    	try:
    		# Get commission rate from consensus
    		commission_rate = self.consensus.get_commission_rate(validator_address)
    		
    		# Get total rewards generated by validator
    		total_rewards = self._calculate_validator_rewards(validator_address)
    		# Calculate commission earnings
    		commission_earnings = int(total_rewards * commission_rate)
    		
    		# Adjust for any commission caps or limits
    		max_commission = self.consensus.get_max_commission(validator_address)
    		if max_commission is not None:
    			commission_earnings = min(commission_earnings, max_commission)
    			
    		return commission_earnings
    		
    	except Exception as e:
    		logger.error(f"Error calculating commission earnings: {e}")
    		return 0
    		 
    def _calculate_delegator_rewards(self, validator_address: str) -> int:
    	"""Calculate total rewards distributed to delegators"""
    	try:
    		total_rewards = self._calculate_validator_rewards(validator_address)
    		
    		# Delegator rewards are total rewards minus commission
    		delegator_rewards = total_rewards - commission_earnings
    		
    		# Ensure non-negative
    		return max(0, delegator_rewards)
    		
    	except Exception as e:
    		logger.error(f"Error calculating delegator rewards: {e}")
    		return 0 
    		
    def _assess_economic_viability(self, staked_amount: int, apr: float, total_stake: int) -> str:
    	"""Comprehensive economic viability assessment"""
    	if total_stake == 0:
    		return 'unknown'
    	stake_percentage = staked_amount / total_stake
    	annual_earnings = staked_amount * apr
    	
    	# Minimum viable earnings threshold (configurable)
    	min_viable_earnings = 1000  # 1000 RXY per year minimum
    	
    	viability_factors = []
    	
    	# Stake percentage factor
    	if stake_percentage < 0.0001:
    		viability_factors.append('very_low_stake')
    	elif stake_percentage < 0.001:
    		viability_factors.append('low_stake')
    		
    	# Earnings factor
    	if annual_earnings < min_viable_earnings:
    		viability_factors.append('low_earnings')
    		
    	# APR factor
    	if apr < 0.05:  # 5% minimum APR
    		viability_factors.append('low_apr')
    		
    	# Determine overall viability
    	if not viability_factors:
    		return 'highly_viable'
    	elif 'very_low_stake' in viability_factors:
    		return 'marginally_viable'
    	else:
    		return 'viable'
    		
    def _calculate_break_even(self, validator_address: str) -> Dict:
    	"""Calculate break-even analysis for validator operation"""
    	try:
    		# Get operational costs (could be configurable or from consensus)
    		operational_costs = {
    		    'server_costs': 200,  # monthly server costs in RXY
    		    'maintenance': 100,   # monthly maintenance
    		    'insurance': 50,      # monthly insurance
    		}
    		monthly_costs = sum(operational_costs.values())
    		annual_costs = monthly_costs * 12
    		
    		# Get expected annual earnings
    		staked_amount = self.consensus.get_stake_amount(validator_address)
    		apr = self._calculate_apr(validator_address, 0)  # 0 for current APR
    		annual_earnings = staked_amount * apr
    		# Calculate break-even
    		if annual_earnings <= annual_costs:
    			return {
    			    'status': 'not_profitable',
    			    'annual_costs': annual_costs,
    			    'annual_earnings': annual_earnings,
    			    'deficit': annual_costs - annual_earnings,
    			    'months_to_breakeven': float('inf')
    			}
    		monthly_earnings = annual_earnings / 12
    		monthly_profit = monthly_earnings - monthly_costs
    		
    		# Time to recover initial investment (simplified)
    		initial_investment = staked_amount
    		months_to_breakeven = initial_investment / monthly_profit
    		
    		return {
    		    'status': 'profitable',
    		    'annual_costs': annual_costs,
    		    'annual_earnings': annual_earnings,
    		    'monthly_profit': monthly_profit,
    		    'months_to_breakeven': round(months_to_breakeven, 1),
    		    'roi_percentage': (annual_earnings - annual_costs) / initial_investment * 100
    		}
    	except Exception as e:
    		logger.error(f"Error calculating break-even: {e}")
    		return {'status': 'calculation_error', 'error': str(e)}
    		
    def _collect_risk_factors(self, validator_address: str) -> Dict:
    	"""Collect comprehensive risk factors for validator"""
    	risk_factors = {}
    	
    	try:
    		# Uptime risk
    		uptime = self.consensus.get_validator_uptime(validator_address)
    		if uptime < 0.95:
    			risk_factors['low_uptime'] = {
    			    'severity': 'high' if uptime < 0.90 else 'medium',
    			    'value': uptime,
    			    'threshold': 0.95
    			}
    			
    		# Slashing history
    		slashing_events = self.consensus.get_slashing_history(validator_address)
    		if slashing_events:
    			risk_factors['slashing_history'] = {
    			    'severity': 'high',
    			    'count': len(slashing_events),
    			    'last_event': max(event['height'] for event in slashing_events)
    			}
    			
    		# Commission change risk
    		commission_history = self.consensus.get_commission_history(validator_address)
    		if commission_history and len(commission_history) > 3:
    			recent_changes = commission_history[-3:]
    			change_rate = statistics.stdev([chg['rate'] for chg in recent_changes])
    			if change_rate > 0.05:
    				risk_factors['volatile_commission'] = {
    				    'severity': 'medium',
    				    'change_rate': change_rate,
    				    'threshold': 0.05
    				}
    		# Delegation concentration risk
    		delegators = self.consensus.get_delegators(validator_address)
    		if delegators:
    			total_delegated = sum(d['amount'] for d in delegators)
    			if total_delegated > 0:
    				top_delegator = max(delegators, key=lambda x: x['amount'])
    				concentration = top_delegator['amount'] / total_delegated
    				if concentration > 0.3:
    					risk_factors['delegation_concentration'] = {
    					    'severity': 'medium',
    					    'concentration': concentration,
    					    'threshold': 0.3
    					    
    					}
    		# Network connectivity risk
    		connectivity = self.consensus.get_connectivity_metrics(validator_address)
    		if connectivity and connectivity.get('latency', 0) > 1000:
    			risk_factors['high_latency'] = {
    			    
    			    'severity': 'medium',
    			    'latency_ms': connectivity['latency'],
    			    'threshold': 1000
    			}
    	except Exception as e:
    		logger.error(f"Error collecting risk factors: {e}")
    		risk_factors['collection_error'] = {'error': str(e)}
    	return risk_factors
    	
    def _calculate_risk_score(self, risk_factors: Dict) -> float:
    	"""Calculate comprehensive risk score from risk factors"""
    	if not risk_factors:
    		return 0.1  # Low risk baseline
    	severity_weights = {
    	    'critical': 1.0,
    	    'high': 0.7,
    	    'medium': 0.4,
    	    'low': 0.2
    	}
    	
    	total_weight = 0
    	weighted_score = 0
    	
    	for factor_name, factor_data in risk_factors.items():
    		severity = factor_data.get('severity', 'medium')
    		weight = severity_weights.get(severity, 0.4)
    		
    		# Additional weight based on factor type
    		if 'slashing' in factor_name:
    			weight *= 1.5
    		elif 'uptime' in factor_name:
    			weight *= 1.2
    		weighted_score += weight
    		total_weight += weight
    		
    	if total_weight == 0:
    		return 0.1
    		
    	# Normalize to 0-1 scale
    	base_score = weighted_score / total_weight
    	
    	# Apply non-linear scaling to emphasize high risks
    	risk_score = min(1.0, base_score * 1.5)
    	return round(risk_score, 4)  

    def _determine_risk_level(self, risk_score: float) -> str:
    	"""Determine risk level with detailed thresholds"""
    	risk_levels = [
    	    (0.20, 'very_low'),
    	    (0.40, 'low'),
    	    (0.60, 'moderate'),
    	    (0.75, 'elevated'),
    	    (0.85, 'high'),
    	    (0.95, 'very_high'),
    	    (1.00, 'extreme')
    	]
    	for threshold, level in risk_levels:
    		if risk_score <= threshold:
    			return level
    			 
    	return 'extreme'    	  	    		         		  
    def _calculate_slashing_impact(self, validator_address: str, risk_score: float) -> Dict:
    	"""Calculate potential slashing impact"""
    	try:
    		staked_amount = self.consensus.get_stake_amount(validator_address)
    		delegated_stake = self.consensus.get_delegated_stake(validator_address)
    		total_stake = staked_amount + delegated_stake
    		
    		# Base slashing percentages from consensus parameters
    		slashing_params = self.consensus.get_slashing_parameters()
    		double_sign_penalty = slashing_params.get('double_sign_penalty', 0.05)  # 5%
    		downtime_penalty = slashing_params.get('downtime_penalty', 0.0001)     # 0.01%
    		
    		# Calculate potential losses
    		potential_losses = {
    		    'double_sign': total_stake * double_sign_penalty * risk_score,
    		    'downtime': total_stake * downtime_penalty * risk_score,
    		    'maximum_possible': total_stake * min(1.0, risk_score * 2)  # Cap at 100%
    		}
    		
    		# Calculate probability-adjusted expected loss
    		expected_loss = (
    		    potential_losses['double_sign'] * 0.1 +  # 10% probability of double sign
    		    potential_losses['downtime'] * 0.3       # 30% probability of downtime
    		)
    		return {
    		    'potential_losses': {k: int(v) for k, v in potential_losses.items()},
    		    'expected_annual_loss': int(expected_loss),
    		    'risk_adjusted_apr_impact': (expected_loss / total_stake) * 100 if total_stake > 0 else 0,
    		    'insurance_coverage_required': int(expected_loss * 1.1)  # 10% buffer
    		}
    		
    	except Exception as e:
    		logger.error(f"Error calculating slashing impact: {e}")
    		return {'error': str(e)}	  
    		
    def _get_risk_trend(self, validator_address: str) -> str:
    	"""Analyze risk trend over time"""
    	try:
    		# Get historical risk scores (last 30 days)
    		historical_scores = self.consensus.get_historical_risk_scores(validator_address, 30)
    		if not historical_scores or len(historical_scores) < 2:
    			return 'insufficient_data'
    		# Calculate trend using linear regression
    		x = list(range(len(historical_scores)))
    		y = historical_scores
    		
    		slope, intercept = statistics.linear_regression(x, y)
    		
    		# Determine trend direction and strength
    		if abs(slope) < 0.001:
    			return 'stable'
    		elif slope > 0.01:
    			return 'increasing_rapidly'
    		elif slope > 0.001:
    			return 'increasing'
    		elif slope < -0.01:
    			return 'decreasing_rapidly'
    		elif slope < -0.001:
    			return 'decreasing'
    		else:
    			return 'stable'
    			
    	except Exception as e:
    		logger.error(f"Error calculating risk trend: {e}")
    		return 'unknown'
    		
    def _get_insurance_coverage(self, validator_address: str) -> Dict:
    	"""Get insurance coverage information"""
    	try:
    		# Check if validator has active insurance
    		insurance_info = self.consensus.get_validator_insurance(validator_address)
    		if not insurance_info:
    			return {
    			    'covered': False,
    			    'recommended_coverage': self._calculate_slashing_impact(validator_address, 0.5)['insurance_coverage_required']
    			    
    			}
    		return {
    		    'covered': True,
    		    'provider': insurance_info.get('provider', 'unknown'),
    		    'coverage_amount': insurance_info.get('coverage_amount', 0),
    		    'premium': insurance_info.get('premium', 0),
    		    'expiration_date': insurance_info.get('expiration_date'),
    		    'deductible': insurance_info.get('deductible', 0),
    		    'coverage_percentage': min(100, (insurance_info.get('coverage_amount', 0) / 
                self.consensus.get_stake_amount(validator_address)) * 100) if self.consensus.get_stake_amount(validator_address) > 0 else 0
    		}
    		
    	except Exception as e:
    		logger.error(f"Error getting insurance coverage: {e}")
    		return {'covered': False, 'error': str(e)}    		
 
    def _get_validation_schedule(self, validator_address: str) -> Dict:
    	"""Get detailed validation schedule"""
    	try:
    		schedule = self.consensus.get_validation_schedule(validator_address)
    		if not schedule:
    			# Calculate estimated schedule based on stake and network parameters
    			total_stake = self.consensus.get_total_stake()
    			validator_stake = self.consensus.get_stake_amount(validator_address)
    			
    			if total_stake > 0 and validator_stake > 0:
    				stake_percentage = validator_stake / total_stake
    				expected_blocks_per_day = (86400 / self.config['block_time_target']) * stake_percentage
    				
    				schedule = {
    				    'estimated_blocks_per_day': round(expected_blocks_per_day, 2),
    				    'estimated_blocks_per_week': round(expected_blocks_per_day * 7, 2),
    				    'validation_frequency': 'variable_based_on_stake',
    				    'next_expected_block': 'unknown'
    				}
    				
    		return schedule or {'status': 'no_schedule_available'}
    		
    	except Exception as e:
    		logger.error(f"Error getting validation schedule: {e}")
    		return {'error': str(e)}
    		
    def _get_network_validator_status(self) -> Dict:
    	"""Get comprehensive network validator status"""
    	try:
    		network_status = self.consensus.get_network_status()
    		if not network_status:
    			# Fallback calculation
    			active_validators = len(self.consensus.get_active_validators())
    			total_stake = self.consensus.get_total_stake()
    			avg_uptime = statistics.mean([
    			    self.consensus.get_validator_uptime(v) 
                for v in self.consensus.get_active_validators()[:10]  # Sample first
    			]) if active_validators > 0 else 0
    			
    			network_status = {
    			    'active_validators': active_validators,
    			    'total_stake': total_stake,
    			    'average_uptime': round(avg_uptime, 4),
    			    'network_health': 'good' if avg_uptime > 0.95 else 'degraded',
    			    'block_time_variance': self._calculate_network_block_time_variance(),
    			    'governance_participation': self.consensus.get_governance_participation_rate()
    			}
    			
    		return network_status
    		
    	except Exception as e:
    	    logger.error(f"Error getting network status: {e}")
    	    return {
    	        'active_validators': 'unknown',
    	        'network_health': 'unknown',
    	        'error': str(e)
    	    }		
  
    def _calculate_network_block_time_variance(self) -> float:
    	"""Calculate network-wide block time variance"""
    	try:
    		# Sample recent blocks for variance calculation
    		recent_blocks = self.blockchain[-100:]  # Last 100 blocks
    		if len(recent_blocks) < 2:
    			return 0.0
    			
    		block_times = []
    		for i in range(1, len(recent_blocks)):
    			time_diff = recent_blocks[i]['timestamp'] - recent_blocks[i-1]['timestamp']
    			block_times.append(time_diff)
    			
    		if len(block_times) < 2:
    			return 0.0
    			
    		variance = statistics.variance(block_times)
    		return round(variance, 4)
    	except Exception as e:
    	    logger.error(f"Error calculating block time variance: {e}")
    	    return 0.0
    	    
    def _get_cached_performance(self, cache_key: str) -> Optional[Dict]:
        """Get cached performance data with validation"""
        try:
        	cached_data = self.database.get(cache_key)
        	if not cached_data or not isinstance(cached_data, dict):
        		return None
        		
        	# Validate cache structure
        	required_fields = ['data', 'timestamp', 'expires_at']
        	if not all(field in cached_data for field in required_fields):
        		return None
        		
        	# Check if cache is expired
        	if time.time() > cached_data['expires_at']:
        		return None
        		
        	# Validate data integrity
        	if 'signature' in cached_data:
        		# Verify cryptographic signature if present
        		data_hash = hashlib.sha256(
        		    json.dumps(cached_data['data'], sort_keys=True).encode()
        		).hexdigest()
        		
        		if not self._verify_cache_signature(data_hash, cached_data['signature']):
        			logger.warning(f"Cache signature verification failed for {cache_key}")
        			return None
        			
        	return cached_data['data']
        	
        except Exception as e:
            logger.warning(f"Cache retrieval error for {cache_key}: {e}")
            return None
            		
    def _verify_cache_signature(self, data_hash: str, signature: str) -> bool:
        """Verify cache data signature"""
        # Implementation would depend on your signing mechanism
        # For now, return True as placeholder
        return True
        
    def _is_performance_data_stale(self, cached_data: Dict) -> bool:
        """Check if performance data is stale with multiple criteria"""
        if not isinstance(cached_data, dict):  
            return True
            
        current_time = time.time()
        cached_time = cached_data.get('timestamp', 0)
        
        # Absolute time-based staleness
        if current_time - cached_time > 300:
        	return True
        	
        # Blockchain progress-based staleness
        cached_height = cached_data.get('block_height', 0)
        current_height = len(self.blockchain) - 1
        if current_height - cached_height > 10:
        	# More than 10 blocks since cache
        	return True
        	
        # Network state-based staleness
        if cached_data.get('network_hash') != self._calculate_network_state_hash():
        	return True
        return False
        
    def _calculate_network_state_hash(self) -> str:
      """Calculate hash of current network state for cache validation"""
      network_state = {
          'height': len(self.blockchain) - 1,
          'total_stake': self.consensus.get_total_stake(),
          'active_validators': len(self.consensus.get_active_validators()),
          'timestamp': int(time.time())
      }
      return hashlib.sha256(json.dumps(network_state, sort_keys=True).encode()).hexdigest()
      
    def _cache_performance_data(self, cache_key: str, data: Dict):
        """Cache performance data with comprehensive metadata"""
        try:
        	cache_data = {
        	    'data': data,
        	    'timestamp': time.time(),
        	    'expires_at': time.time() + 300,  # 5 minutes
        	    'block_height': len(self.blockchain) - 1,
        	    'network_hash': self._calculate_network_state_hash(),
        	    'node_id': getattr(self, 'node_id', 'unknown'),
        	    'version': '1.0'
        	}
        	
        	# Add cryptographic signature for data integrity
        	data_hash = hashlib.sha256(
        	    json.dumps(data, sort_keys=True).encode()
        	).hexdigest()
        	cache_data['signature'] = self._sign_cache_data(data_hash)
        	
        	self.database.put(cache_key, cache_data)
        	
        except Exception as e:
        	logger.error(f"Failed to cache performance data: {e}")
        	            
    def _sign_cache_data(self, data_hash: str) -> str:
        """Sign cache data for integrity verification"""
        # Implementation would depend on your signing mechanism
        # Return placeholder for now
        return f"signed_{data_hash[:16]}"            
      
    def _sign_validator_info(self, data: Dict) -> Dict:
    	"""Sign validator information for data integrity."""
    	try:
    		# Create hash of the data
    		data_hash = hashlib.sha256(
    		    json.dumps(data, sort_keys=True).encode()
    		).hexdigest()
    		
    		# Sign with node key if available
    		signature = None
    		if hasattr(self, 'node_key'):
    			signature = self.node_key.sign(data_hash.encode()).hex()
    			return {
    			    'data_hash': data_hash,
    			    'signature': signature,
    			    'signing_node': getattr(self, 'node_id', 'unknown'),
    			    'timestamp': int(time.time())
    			}
    	except Exception as e:
    		logger.warning(f"Failed to sign validator info: {e}")
    		return {'error': f'signing_failed: {str(e)}'}
    		        
    def _cache_validator_info(self, address: str, data: Dict):
    	"""Cache validator information with expiration."""
    	try:
    		cache_data = {
    		    'data': data,
    		    'timestamp': time.time(),
    		    'expires_at': time.time() + 300,
    		    'height': len(self.blockchain) - 1
    		}
    		cache_key = f"validator_info_{address}"
    		self.database.put(cache_key, cache_data)
    	except Exception as e:
    		logger.warning(f"Failed to cache validator info: {e}")
    		
# Additional helper methods would be implemented here for:
# - _calculate_performance_score
# - _get_reliability_rating
# - _get_efficiency_rating
# - _calculate_apy
# - _calculate_commission_earnings
# - _calculate_delegator_rewards
# - _assess_economic_viability
# - _calculate_break_even
# - _collect_risk_factors
# - _calculate_risk_score
# - _determine_risk_level
# - _calculate_slashing_impact
# - _get_risk_trend
# - _get_insurance_coverage
# - And many more...    		
    		
    def _get_data_freshness(self) -> str:
    	"""Get data freshness indicator."""
    	height = len(self.blockchain) - 1
    	if height < 10:
    		return 'very_fresh'
    	elif height < 100:
    		return 'fresh'
    	elif height < 1000:
    		return 'recent'
    	else:
    		return 'established'
    	
           	
    	
def calculate_mining_reward(height: int, base_reward: int = 50, halving_interval: int = 210000) -> int:
    """Calculate mining reward at given height"""
    halvings = height // halving_interval
    reward = base_reward >> halvings
    return max(reward, 1)

# Example usage
if __name__ == "__main__":
    # Create mainnet instance
    rayonix = RayonixCoin("mainnet")
    
    try:
        # Start network and mining
        rayonix.connect_to_network()
        rayonix.start_mining()
        
        # Display blockchain info
        info = rayonix.get_blockchain_info()
        print(f"RAYONIX Blockchain Info:")
        print(f"  Height: {info['height']}")
        print(f"  Total Supply: {info['total_supply']} RXY")
        print(f"  Circulating Supply: {info['circulating_supply']} RXY")
        print(f"  Current Reward: {info['block_reward']} RXY")
        print(f"  Difficulty: {info['difficulty']}")
        
        # Create a transaction (example)
        if rayonix.wallet and rayonix.wallet.addresses:
            address = list(rayonix.wallet.addresses.keys())[0]
            balance = rayonix.get_balance(address)
            print(f"Wallet Balance: {balance} RXY")
            
            # Send transaction if we have funds
            if balance > 10:
                tx = rayonix.create_transaction(
                    address,
                    "rx1recipientaddressxxxxxxxxxxxxxx",
                    10,
                    fee=1
                )
                if tx:
                    print(f"Transaction created: {rayonix._calculate_transaction_hash(tx)[:16]}...")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            
    finally:
        rayonix.stop_mining()
        rayonix.disconnect_from_network()
        rayonix.close()