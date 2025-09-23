# blockchain/utils/genesis.py
import time
import hashlib
import json
from typing import Dict, Any, Optional

from blockchain.models.block import Block, BlockHeader
from utxo_system.models.transaction import Transaction

class GenesisBlockGenerator:
    """Generates genesis blocks for different network configurations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def generate_genesis_block(self, custom_config: Optional[Dict[str, Any]] = None) -> Block:
        """Generate genesis block with custom configuration"""
        config = self._merge_configs(custom_config)
        
        # Create premine transaction
        premine_tx = self._create_premine_transaction(config)
        
        # Create genesis block header
        header = self._create_genesis_header(config, [premine_tx])
        
        # Create genesis block
        genesis_block = Block(
            header=header,
            transactions=[premine_tx],
            hash=header.calculate_hash(),
            chainwork=1,
            size=self._calculate_block_size(header, [premine_tx])
        )
        
        # Sign genesis block if private key provided
        if config.get('private_key'):
            genesis_block = self._sign_genesis_block(genesis_block, config['private_key'])
        
        return genesis_block
    
    def _merge_configs(self, custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge custom config with default config"""
        default_config = {
            'premine_amount': self.config.get('genesis_premine', 1000000000000000),
            'foundation_address': self.config.get('foundation_address', 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX'),
            'block_reward': self.config.get('block_reward', 50),
            'network_id': self.config.get('network_id', 1),
            'timestamp': int(time.time()),
            'validator': 'genesis',
            'version': 1,
            'difficulty': 1,
            'nonce': 0,
            'block_time_target': self.config.get('block_time_target', 30),
            'max_supply': self.config.get('max_supply', 21000000),
            'developer_fee_percent': self.config.get('developer_fee_percent', 0.05)
        }
        
        if custom_config:
            default_config.update(custom_config)
        
        return default_config
    
    def _create_premine_transaction(self, config: Dict[str, Any]) -> Transaction:
        """Create premine transaction"""
        output = TransactionOutput(
            address=config['foundation_address'],
            amount=config['premine_amount'],
            locktime=0
        )
        
        return Transaction(
            inputs=[],
            outputs=[output],
            locktime=0,
            version=1,
            timestamp=config['timestamp']
        )
    
    def _create_genesis_header(self, config: Dict[str, Any], transactions: List[Transaction]) -> BlockHeader:
        """Create genesis block header"""
        from blockchain.utils.block_calculations import calculate_merkle_root
        
        tx_hashes = [tx.hash for tx in transactions]
        merkle_root = calculate_merkle_root(tx_hashes)
        
        return BlockHeader(
            version=config['version'],
            height=0,
            previous_hash='0' * 64,
            merkle_root=merkle_root,
            timestamp=config['timestamp'],
            difficulty=config['difficulty'],
            nonce=config['nonce'],
            validator=config['validator'],
            extra_data={
                'network_id': config['network_id'],
                'premine_amount': config['premine_amount'],
                'foundation_address': config['foundation_address'],
                'config_hash': self._calculate_config_hash(config)
            }
        )
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for integrity checking"""
        config_data = {k: v for k, v in config.items() if k not in ['private_key']}
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _calculate_block_size(self, header: BlockHeader, transactions: List[Transaction]) -> int:
        """Calculate genesis block size"""
        header_size = len(header.to_bytes())
        transactions_size = sum(len(tx.to_bytes()) for tx in transactions)
        return header_size + transactions_size
    
    def _sign_genesis_block(self, block: Block, private_key: str) -> Block:
        """Sign genesis block with provided private key"""
        try:
            from blockchain.production.block_signing import BlockSigner
            signer = BlockSigner(self.config)
            
            # Convert private key from hex to bytes
            private_key_bytes = bytes.fromhex(private_key)
            
            # Sign the block
            signature = signer.sign_block(block, private_key_bytes)
            if signature:
                block.header.signature = signature.hex()
            
            return block
            
        except Exception as e:
            print(f"Failed to sign genesis block: {e}")
            return block
    
    def validate_genesis_block(self, genesis_block: Block, expected_config: Dict[str, Any]) -> bool:
        """Validate genesis block matches expected configuration"""
        try:
            # Check basic structure
            if genesis_block.header.height != 0:
                return False
            
            if genesis_block.header.previous_hash != '0' * 64:
                return False
            
            # Check transactions
            if len(genesis_block.transactions) != 1:
                return False
            
            premine_tx = genesis_block.transactions[0]
            if len(premine_tx.inputs) != 0:
                return False
            
            if len(premine_tx.outputs) != 1:
                return False
            
            # Check premine amount
            expected_amount = expected_config.get('premine_amount', 1000000000000000)
            if premine_tx.outputs[0].amount != expected_amount:
                return False
            
            # Check foundation address
            expected_address = expected_config.get('foundation_address')
            if expected_address and premine_tx.outputs[0].address != expected_address:
                return False
            
            # Verify block hash
            if genesis_block.hash != genesis_block.header.calculate_hash():
                return False
            
            return True
            
        except Exception as e:
            print(f"Genesis block validation failed: {e}")
            return False
    
    def get_genesis_config_template(self) -> Dict[str, Any]:
        """Get template for genesis block configuration"""
        return {
            'premine_amount': {
                'description': 'Initial coin supply for foundation',
                'type': 'integer',
                'default': 1000000000000000,
                'min': 0,
                'max': 1000000000000000000
            },
            'foundation_address': {
                'description': 'Address to receive premined coins',
                'type': 'string',
                'default': 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX',
                'pattern': '^[A-Z0-9]{34}$'
            },
            'network_id': {
                'description': 'Unique network identifier',
                'type': 'integer',
                'default': 1,
                'min': 1,
                'max': 65535
            },
            'difficulty': {
                'description': 'Initial mining difficulty',
                'type': 'integer',
                'default': 1,
                'min': 1,
                'max': 1000000
            },
            'block_time_target': {
                'description': 'Target time between blocks in seconds',
                'type': 'integer',
                'default': 30,
                'min': 1,
                'max': 600
            },
            'private_key': {
                'description': 'Private key for signing genesis block (optional)',
                'type': 'string',
                'pattern': '^[a-fA-F0-9]{64}$'
            }
        }