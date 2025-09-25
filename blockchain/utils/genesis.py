# blockchain/utils/genesis.py
import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import secrets

from blockchain.models.block import Block, BlockHeader
from utxo_system.models.transaction import Transaction, TransactionOutput


class GenesisBlockGenerator:
    """Generates and validates genesis blocks for different network configurations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validation_cache: Dict[str, bool] = {}
    
    def generate_genesis_block(self, custom_config: Optional[Dict[str, Any]] = None) -> Block:
        """Generate genesis block with custom configuration and comprehensive validation"""
        try:
            config = self._merge_configs(custom_config)
            self._validate_genesis_config(config)
            
            # Create premine transaction with enhanced metadata
            premine_tx = self._create_premine_transaction(config)
            
            # Create genesis block header with additional security features
            header = self._create_genesis_header(config, [premine_tx])
            
            # Create genesis block with comprehensive metadata
            genesis_block = Block(
                header=header,
                transactions=[premine_tx],
                hash=header.calculate_hash(),
                chainwork=self._calculate_initial_chainwork(config),
                size=self._calculate_block_size(header, [premine_tx]),
                weight=self._calculate_block_weight(header, [premine_tx]),
                validation_status="genesis",
                received_timestamp=config['timestamp'],
                propagation_time=0
            )
            
            # Add cryptographic signatures and proofs
            genesis_block = self._enhance_genesis_security(genesis_block, config)
            
            # Cache validation result
            cache_key = self._generate_validation_cache_key(genesis_block)
            self._validation_cache[cache_key] = True
            
            return genesis_block
            
        except Exception as e:
            raise GenesisGenerationError(f"Failed to generate genesis block: {e}") from e
    
    def _merge_configs(self, custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge custom config with default config with comprehensive validation"""
        default_config = {
            'premine_amount': self.config.get('genesis_premine', 1000000000000000),
            'foundation_address': self.config.get('foundation_address', 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX'),
            'block_reward': self.config.get('block_reward', 50),
            'network_id': self.config.get('network_id', 1),
            'timestamp': int(time.time()),
            'validator': 'genesis',
            'version': 1,
            'difficulty': 1,
            'nonce': secrets.randbelow(1000000),  # Randomized nonce for security
            'block_time_target': self.config.get('block_time_target', 30),
            'max_supply': self.config.get('max_supply', 21000000),
            'developer_fee_percent': self.config.get('developer_fee_percent', 0.05),
            'genesis_description': self.config.get('genesis_description', 'Initial blockchain genesis block'),
            'consensus_algorithm': self.config.get('consensus_algorithm', 'pow'),
            'security_level': self.config.get('security_level', 'high'),
            'genesis_metadata': self.config.get('genesis_metadata', {})
        }
        
        if custom_config:
            self._validate_custom_config(custom_config)
            default_config.update(custom_config)
        
        return default_config
    
    def _validate_custom_config(self, config: Dict[str, Any]) -> None:
        """Validate custom configuration parameters"""
        template = self.get_genesis_config_template()
        
        for key, value in config.items():
            if key not in template:
                raise ValueError(f"Invalid configuration parameter: {key}")
            
            param_config = template[key]
            param_type = param_config['type']
            
            # Type validation
            if param_type == 'integer':
                if not isinstance(value, int):
                    raise ValueError(f"Parameter {key} must be integer")
                if 'min' in param_config and value < param_config['min']:
                    raise ValueError(f"Parameter {key} must be >= {param_config['min']}")
                if 'max' in param_config and value > param_config['max']:
                    raise ValueError(f"Parameter {key} must be <= {param_config['max']}")
            
            elif param_type == 'string':
                if not isinstance(value, str):
                    raise ValueError(f"Parameter {key} must be string")
                if 'pattern' in param_config:
                    import re
                    if not re.match(param_config['pattern'], value):
                        raise ValueError(f"Parameter {key} does not match required pattern")
    
    def _validate_genesis_config(self, config: Dict[str, Any]) -> None:
        """Validate genesis configuration for consistency and security"""
        if config['premine_amount'] > config['max_supply']:
            raise ValueError("Premine amount cannot exceed max supply")
        
        if config['premine_amount'] <= 0:
            raise ValueError("Premine amount must be positive")
        
        if config['network_id'] == 0:
            raise ValueError("Network ID 0 is reserved")
        
        if config['difficulty'] <= 0:
            raise ValueError("Difficulty must be positive")
    
    def _create_premine_transaction(self, config: Dict[str, Any]) -> Transaction:
        """Create premine transaction with enhanced metadata and security"""
        coinbase_input = TransactionInput(
            tx_hash="0" * 64,
            output_index=-1,
            sequence=0xFFFFFFFF,
            signature=None,
            public_key=None,
            address="coinbase",
            witness=[],
            script_sig="genesis_coinbase"
        )
        output = TransactionOutput(
            address=config['foundation_address'],
            amount=config['premine_amount'],
            locktime=0
        )
        # Create transaction with the coinbase input
        premine_tx = Transaction(
            inputs=[coinbase_input],
            outputs=[output],
            locktime=0,
            version=1
        )
        premine_tx.metadata = {
            'network_id': config['network_id'],
            'premine_amount': config['premine_amount'],
            'block_height': 0,
            'consensus_algorithm': config['consensus_algorithm'],
            'timestamp': config['timestamp'],
            'premine_info': {
                'type': 'foundation',
                'network_id': config['network_id'],
                'creation_timestamp': config['timestamp'],
                'premine_type': 'foundation'
            },
            'is_genesis': True
        }
        return premine_tx
    
    def _create_genesis_header(self, config: Dict[str, Any], transactions: List[Transaction]) -> BlockHeader:
        """Create genesis block header with enhanced security features"""
        from blockchain.utils.block_calculations import calculate_merkle_root
        
        tx_hashes = [tx.hash for tx in transactions]
        merkle_root = calculate_merkle_root(tx_hashes)
        
        # Calculate additional security hashes
        config_hash = self._calculate_config_hash(config)
        security_hash = self._calculate_security_hash(config, merkle_root)
        
        return BlockHeader(
            version=config['version'],
            height=0,
            previous_hash='0' * 64,
            merkle_root=merkle_root,
            timestamp=config['timestamp'],
            difficulty=config['difficulty'],
            nonce=config['nonce'],
            validator='genesis',
            signature=None,
            extra_data={
                'network_id': config['network_id'],
                'premine_amount': config['premine_amount'],
                'foundation_address': config['foundation_address'],
                'config_hash': config_hash,
                'security_hash': security_hash,
                'genesis_description': config['genesis_description'],
                'consensus_algorithm': config['consensus_algorithm'],
                'block_time_target': config['block_time_target'],
                'max_supply': config['max_supply'],
                'developer_fee_percent': config['developer_fee_percent'],
                'creation_timestamp': config['timestamp'],
                'metadata_hash': self._calculate_metadata_hash(config.get('genesis_metadata', {})),
                'is_genesis': True
            }
        )
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for integrity checking with enhanced security"""
        config_data = {k: v for k, v in config.items() if k not in ['private_key', 'signature']}
        config_str = json.dumps(config_data, sort_keys=True, separators=(',', ':'))
        
        # Double hash for additional security
        first_hash = hashlib.sha256(config_str.encode()).digest()
        return hashlib.sha256(first_hash).hexdigest()
    
    def _calculate_security_hash(self, config: Dict[str, Any], merkle_root: str) -> str:
        """Calculate additional security hash for genesis block validation"""
        security_data = {
            'network_id': config['network_id'],
            'timestamp': config['timestamp'],
            'merkle_root': merkle_root,
            'premine_amount': config['premine_amount'],
            'difficulty': config['difficulty']
        }
        security_str = json.dumps(security_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha3_256(security_str.encode()).hexdigest()
    
    def _calculate_metadata_hash(self, metadata: Dict[str, Any]) -> str:
        """Calculate hash of genesis metadata"""
        if not metadata:
            return '0' * 64
        
        metadata_str = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
        return hashlib.blake2s(metadata_str.encode()).hexdigest()
    
    def _calculate_initial_chainwork(self, config: Dict[str, Any]) -> int:
        """Calculate initial chainwork based on difficulty"""
        # Simple chainwork calculation for genesis block
        return max(1, config['difficulty'] * 1000)
    
    def _calculate_block_size(self, header: BlockHeader, transactions: List[Transaction]) -> int:
        """Calculate genesis block size with comprehensive serialization"""
        try:
            header_size = len(json.dumps(asdict(header), separators=(',', ':')).encode('utf-8'))
            transactions_size = sum(len(json.dumps(asdict(tx), separators=(',', ':')).encode('utf-8')) 
                                  for tx in transactions)
            return header_size + transactions_size
        except Exception:
            # Fallback calculation
            return 1024  # Conservative estimate for genesis block
    
    def _calculate_block_weight(self, header: BlockHeader, transactions: List[Transaction]) -> int:
        """Calculate block weight for fee calculation compatibility"""
        base_size = self._calculate_block_size(header, transactions)
        total_size = base_size + sum(len(tx.outputs) * 100 for tx in transactions)  # Estimate output weight
        return total_size
    
    def _enhance_genesis_security(self, block: Block, config: Dict[str, Any]) -> Block:
        """Add cryptographic signatures and security enhancements to genesis block"""
        
        # Sign genesis block if private key provided
        if config.get('private_key'):
            block = self._sign_genesis_block(block, config['private_key'])
        
        # Add additional security metadata
        block.security_metadata = {
            'genesis_validated': True,
            'config_integrity_hash': config.get('config_hash', ''),
            'generation_timestamp': config['timestamp'],
            'security_level': config.get('security_level', 'high')
        }
        
        return block
    
    def _sign_genesis_block(self, block: Block, private_key: str) -> Block:
        """Sign genesis block with provided private key using enhanced security"""
        try:
            from blockchain.production.block_signing import BlockSigner
            signer = BlockSigner(self.config)
            
            # Validate private key format
            if len(private_key) != 64:
                raise ValueError("Private key must be 64 characters hex string")
            
            # Convert private key from hex to bytes
            private_key_bytes = bytes.fromhex(private_key)
            
            # Sign the block with additional security context
            signature = signer.sign_block(block, private_key_bytes)
            if signature:
                block.header.signature = signature.hex()
                block.security_metadata['signed'] = True
                block.security_metadata['signature_timestamp'] = int(time.time())
            else:
                block.security_metadata['signed'] = False
            
            return block
            
        except Exception as e:
            print(f"Warning: Failed to sign genesis block: {e}")
            block.security_metadata['signed'] = False
            return block
    
    def validate_genesis_block(self, genesis_block: Block, expected_config: Dict[str, Any]) -> bool:
        """Comprehensive validation of genesis block against expected configuration"""
        try:
            # Check cache first
            cache_key = self._generate_validation_cache_key(genesis_block)
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]
            
            # Basic structure validation
            if genesis_block.header.height != 0:
                return False
            
            if genesis_block.header.previous_hash != '0' * 64:
                return False
            
            # Transaction validation
            if len(genesis_block.transactions) != 1:
                return False
            
            premine_tx = genesis_block.transactions[0]
            if len(premine_tx.inputs) != 0:
                return False
            
            if len(premine_tx.outputs) != 1:
                return False
            
            # Configuration validation
            expected_amount = expected_config.get('premine_amount', 1000000000000000)
            if premine_tx.outputs[0].amount != expected_amount:
                return False
            
            expected_address = expected_config.get('foundation_address')
            if expected_address and premine_tx.outputs[0].address != expected_address:
                return False
            
            # Cryptographic validation
            if genesis_block.hash != genesis_block.header.calculate_hash():
                return False
            
            # Security hash validation
            extra_data = genesis_block.header.extra_data or {}
            security_hash = extra_data.get('security_hash')
            if security_hash:
                expected_security_hash = self._calculate_security_hash(
                    expected_config, 
                    genesis_block.header.merkle_root
                )
                if security_hash != expected_security_hash:
                    return False
            
            # Config hash validation
            config_hash = extra_data.get('config_hash')
            if config_hash:
                expected_config_hash = self._calculate_config_hash(expected_config)
                if config_hash != expected_config_hash:
                    return False
            
            # Cache successful validation
            self._validation_cache[cache_key] = True
            return True
            
        except Exception as e:
            print(f"Genesis block validation failed: {e}")
            return False
    
    def _generate_validation_cache_key(self, genesis_block: Block) -> str:
        """Generate cache key for genesis block validation results"""
        block_data = f"{genesis_block.hash}_{genesis_block.header.timestamp}"
        return hashlib.md5(block_data.encode()).hexdigest()
    
    def get_genesis_config_template(self) -> Dict[str, Any]:
        """Get comprehensive template for genesis block configuration"""
        return {
            'premine_amount': {
                'description': 'Initial coin supply for foundation',
                'type': 'integer',
                'default': 1000000000000000,
                'min': 1,
                'max': 1000000000000000000,
                'security_level': 'high'
            },
            'foundation_address': {
                'description': 'Address to receive premined coins',
                'type': 'string',
                'default': 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX',
                'pattern': '^[A-Z0-9]{34}$',
                'security_level': 'high'
            },
            'network_id': {
                'description': 'Unique network identifier',
                'type': 'integer',
                'default': 1,
                'min': 1,
                'max': 65535,
                'security_level': 'medium'
            },
            'difficulty': {
                'description': 'Initial mining difficulty',
                'type': 'integer',
                'default': 1,
                'min': 1,
                'max': 1000000,
                'security_level': 'medium'
            },
            'block_time_target': {
                'description': 'Target time between blocks in seconds',
                'type': 'integer',
                'default': 30,
                'min': 1,
                'max': 600,
                'security_level': 'medium'
            },
            'private_key': {
                'description': 'Private key for signing genesis block (optional)',
                'type': 'string',
                'pattern': '^[a-fA-F0-9]{64}$',
                'security_level': 'critical'
            },
            'max_supply': {
                'description': 'Maximum coin supply for the network',
                'type': 'integer',
                'default': 21000000,
                'min': 1000000,
                'max': 1000000000000000000,
                'security_level': 'high'
            },
            'consensus_algorithm': {
                'description': 'Consensus algorithm to use',
                'type': 'string',
                'default': 'pow',
                'options': ['pow', 'pos', 'dpos', 'poa'],
                'security_level': 'high'
            },
            'genesis_description': {
                'description': 'Description of the genesis block',
                'type': 'string',
                'default': 'Initial blockchain genesis block',
                'max_length': 1000,
                'security_level': 'low'
            }
        }
    
    def generate_genesis_documentation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive documentation for the genesis block"""
        genesis_block = self.generate_genesis_block(config)
        
        return {
            'genesis_block': {
                'hash': genesis_block.hash,
                'timestamp': genesis_block.header.timestamp,
                'difficulty': genesis_block.header.difficulty,
                'merkle_root': genesis_block.header.merkle_root
            },
            'configuration': config,
            'validation': {
                'is_valid': self.validate_genesis_block(genesis_block, config),
                'validation_timestamp': int(time.time()),
                'security_level': config.get('security_level', 'high')
            },
            'metadata': {
                'generator_version': '1.0.0',
                'generation_timestamp': int(time.time()),
                'network_parameters': {
                    'max_supply': config.get('max_supply'),
                    'block_time_target': config.get('block_time_target'),
                    'consensus_algorithm': config.get('consensus_algorithm')
                }
            }
        }
    
    def cleanup_validation_cache(self, max_age_seconds: int = 3600) -> None:
        """Clean up old validation cache entries"""
        current_time = time.time()
        keys_to_remove = []
        
        # Simple cache cleanup (in production, use proper TTL cache)
        if current_time % 300 < 30:  # Cleanup every ~5 minutes
            keys_to_remove = list(self._validation_cache.keys())[:10]  # Remove first 10 entries
            
        for key in keys_to_remove:
            del self._validation_cache[key]


class GenesisGenerationError(Exception):
    """Exception raised for genesis block generation errors"""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        self.timestamp = int(time.time())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging"""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'original_error': str(self.original_error) if self.original_error else None,
            'timestamp': self.timestamp
        }


# Utility functions for external use
def create_genesis_block(config: Dict[str, Any]) -> Block:
    """Convenience function to create a genesis block"""
    generator = GenesisBlockGenerator(config)
    return generator.generate_genesis_block()

def validate_genesis_block(genesis_block: Block, expected_config: Dict[str, Any]) -> bool:
    """Convenience function to validate a genesis block"""
    generator = GenesisBlockGenerator({})
    return generator.validate_genesis_block(genesis_block, expected_config)

def get_genesis_config_template() -> Dict[str, Any]:
    """Convenience function to get genesis configuration template"""
    generator = GenesisBlockGenerator({})
    return generator.get_genesis_config_template()