# main.py
import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable, Type, Set
from pathlib import Path
import logging
import signal
import threading
from dataclasses import asdict, dataclass
import readline
import uuid
import aiohttp
from aiohttp import web
import jsonrpcserver
from jsonrpcserver import method, async_dispatch
from jsonrpcserver.exceptions import InvalidParams

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rayonix_node.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RayonixNode")

# Import the config system
from config import ConfigManager, init_config, get_config, ConfigFormat, ConfigSource

# Import rayonix_coin instead of blockchain
from rayonix_coin import RayonixCoin, create_rayonix_network, validate_rayonix_address
from wallet import RayonixWallet, WalletConfig, WalletType, AddressType, create_new_wallet
from p2p_network import AdvancedP2PNetwork, NodeConfig, NetworkType, ProtocolType, MessageType
from smart_contract import ContractManager, SmartContract
from database import AdvancedDatabase, DatabaseConfig
from merkle import MerkleTree

@dataclass
class NodeDependencies:
    """Container for node dependencies to enable dependency injection"""
    config_manager: ConfigManager
    rayonix_coin: RayonixCoin
    wallet: Optional[RayonixWallet] = None
    network: Optional[AdvancedP2PNetwork] = None
    contract_manager: Optional[ContractManager] = None
    database: Optional[AdvancedDatabase] = None

class RayonixAPIServer:
    """API server for RAYONIX blockchain node with JSON-RPC and REST endpoints"""
    
    def __init__(self, node: 'RayonixNode', host: str = "127.0.0.1", port: int = 8545):
        self.node = node
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all API routes"""
        # JSON-RPC endpoint
        self.app.router.add_post('/jsonrpc', self.handle_jsonrpc)
        
        # REST API endpoints
        self.app.router.add_get('/api/v1/block/{identifier}', self.get_block)
        self.app.router.add_get('/api/v1/transaction/{tx_hash}', self.get_transaction)
        self.app.router.add_get('/api/v1/address/{address}', self.get_address_info)
        self.app.router.add_get('/api/v1/mempool', self.get_mempool)
        self.app.router.add_post('/api/v1/broadcast', self.broadcast_transaction)
        self.app.router.add_get('/api/v1/network', self.get_network_info)
        self.app.router.add_get('/api/v1/status', self.get_node_status)
    
    async def handle_jsonrpc(self, request):
        """Handle JSON-RPC requests"""
        request_data = await request.text()
        response = await async_dispatch(request_data, context=self.node)
        return web.Response(text=response, content_type="application/json")
    
    async def get_block(self, request):
        """REST endpoint to get block information"""
        identifier = request.match_info['identifier']
        
        try:
            # Try to parse as height first
            try:
                height = int(identifier)
                block = self.node.rayonix_coin.get_block(height)
            except ValueError:
                # Treat as hash
                block = self.node.rayonix_coin.get_block(identifier)
            
            if not block:
                return web.json_response({"error": "Block not found"}, status=404)
            
            return web.json_response(block)
        except Exception as e:
            logger.error(f"Error getting block: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_transaction(self, request):
        """REST endpoint to get transaction information"""
        tx_hash = request.match_info['tx_hash']
        
        try:
            transaction = self.node.rayonix_coin.get_transaction(tx_hash)
            if not transaction:
                return web.json_response({"error": "Transaction not found"}, status=404)
            
            return web.json_response(transaction)
        except Exception as e:
            logger.error(f"Error getting transaction: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_address_info(self, request):
        """REST endpoint to get address information"""
        address = request.match_info['address']
        
        try:
            # Validate address
            if not validate_rayonix_address(address):
                return web.json_response({"error": "Invalid address"}, status=400)
            
            # Get balance
            balance = self.node.rayonix_coin.get_balance(address)
            
            # Get transaction history (simplified implementation)
            transactions = self.node.rayonix_coin.get_address_transactions(address, limit=50)
            
            response = {
                "address": address,
                "balance": balance,
                "transaction_count": len(transactions),
                "transactions": transactions
            }
            
            return web.json_response(response)
        except Exception as e:
            logger.error(f"Error getting address info: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_mempool(self, request):
        """REST endpoint to get mempool information"""
        try:
            mempool = self.node.rayonix_coin.mempool
            return web.json_response({
                "count": len(mempool),
                "transactions": mempool[:100]  # Limit to first 100
            })
        except Exception as e:
            logger.error(f"Error getting mempool: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def broadcast_transaction(self, request):
        """REST endpoint to broadcast a transaction"""
        try:
            data = await request.json()
            transaction = data.get('transaction')
            
            if not transaction:
                return web.json_response({"error": "Transaction data required"}, status=400)
            
            # Validate transaction
            if not self.node.rayonix_coin._validate_transaction(transaction):
                return web.json_response({"error": "Invalid transaction"}, status=400)
            
            # Add to mempool
            self.node.rayonix_coin._add_to_mempool(transaction)
            
            # Broadcast to network if available
            if self.node.network:
                await self.node._broadcast_transaction(transaction)
            
            return web.json_response({
                "success": True,
                "txid": transaction.get('hash')
            })
        except Exception as e:
            logger.error(f"Error broadcasting transaction: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_network_info(self, request):
        """REST endpoint to get network information"""
        try:
            if not self.node.network:
                return web.json_response({"error": "Network not enabled"}, status=503)
            
            info = self.node.network.get_metrics() or {}
            return web.json_response(info)
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_node_status(self, request):
        """REST endpoint to get node status"""
        try:
            status = {
                'running': self.node.running,
                'network': self.node.get_config_value('network.network_type', 'mainnet'),
                'network_id': self.node.get_config_value('network.network_id', 1),
                'block_height': len(self.node.rayonix_coin.blockchain),
                'connected_peers': self.node.sync_state['peers_connected'],
                'syncing': self.node.sync_state['syncing'],
                'consensus': self.node.get_config_value('consensus.consensus_type', 'pos'),
                'wallet_loaded': self.node.wallet is not None,
                'api_enabled': True,
                'api_port': self.port
            }
            
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting node status: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def start(self):
        """Start the API server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            logger.info(f"API server started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    async def stop(self):
        """Stop the API server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("API server stopped")

# JSON-RPC methods
@method
async def sendrawtransaction(context, hex_tx: str) -> Dict:
    """JSON-RPC method to send a raw transaction"""
    try:
        # Decode hex transaction
        try:
            tx_data = json.loads(bytes.fromhex(hex_tx).decode('utf-8'))
        except:
            return {"error": "Invalid transaction hex encoding"}
        
        # Validate transaction
        if not context.rayonix_coin._validate_transaction(tx_data):
            return {"error": "Invalid transaction"}
        
        # Add to mempool
        context.rayonix_coin._add_to_mempool(tx_data)
        
        # Broadcast to network if available
        if context.network:
            await context._broadcast_transaction(tx_data)
        
        return {"result": tx_data.get('hash'), "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def createrawtransaction(context, inputs: List[Dict], outputs: List[Dict]) -> Dict:
    """JSON-RPC method to create a raw transaction"""
    try:
        # Validate inputs
        if not inputs or not outputs:
            return {"error": "Inputs and outputs required"}
        
        # Create transaction
        transaction = {
            "version": 1,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": int(time.time()),
            "locktime": 0
        }
        
        # Calculate hash
        import hashlib
        tx_json = json.dumps(transaction, sort_keys=True).encode('utf-8')
        transaction["hash"] = hashlib.sha256(tx_json).hexdigest()
        
        # Return hex encoded transaction
        hex_tx = tx_json.hex()
        return {"result": hex_tx, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def signrawtransactionwithwallet(context, hex_tx: str) -> Dict:
    """JSON-RPC method to sign a transaction with wallet"""
    try:
        if not context.wallet:
            return {"error": "No wallet loaded"}
        
        # Decode hex transaction
        try:
            tx_data = json.loads(bytes.fromhex(hex_tx).decode('utf-8'))
        except:
            return {"error": "Invalid transaction hex encoding"}
        
        # Sign transaction (simplified implementation)
        # In a real implementation, this would properly sign each input
        signed_tx = context.wallet.sign_transaction(tx_data)
        
        if not signed_tx:
            return {"error": "Failed to sign transaction"}
        
        # Return hex encoded signed transaction
        signed_hex = json.dumps(signed_tx, sort_keys=True).encode('utf-8').hex()
        return {"result": signed_hex, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getbestblockhash(context) -> Dict:
    """JSON-RPC method to get best block hash"""
    try:
        blockchain = context.rayonix_coin.blockchain
        if not blockchain:
            return {"result": "0" * 64, "error": None}
        
        last_block = blockchain[-1]
        return {"result": last_block.get('hash', "0" * 64), "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getblockcount(context) -> Dict:
    """JSON-RPC method to get block count"""
    try:
        return {"result": len(context.rayonix_coin.blockchain), "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getbalance(context, address: str) -> Dict:
    """JSON-RPC method to get address balance"""
    try:
        # Validate address
        if not validate_rayonix_address(address):
            return {"error": "Invalid address"}
        
        balance = context.rayonix_coin.get_balance(address)
        return {"result": balance, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getchaintips(context) -> Dict:
    """JSON-RPC method to get chain tips"""
    try:
        # In a simple implementation, we only have one chain
        blockchain = context.rayonix_coin.blockchain
        if not blockchain:
            return {"result": [], "error": None}
        
        last_block = blockchain[-1]
        tip = {
            "height": len(blockchain) - 1,
            "hash": last_block.get('hash', "0" * 64),
            "branchlen": 0,
            "status": "active"
        }
        
        return {"result": [tip], "error": None}
    except Exception as e:
        return {"error": str(e)}

class RayonixNode:
    """Complete RAYONIX blockchain node using rayonix_coin.py as backend"""
    
    def __init__(self, dependencies: Optional[NodeDependencies] = None):
        # Use provided dependencies or create empty ones
        if dependencies:
            self.deps = dependencies
            self.config_manager = dependencies.config_manager
            self.rayonix_coin = dependencies.rayonix_coin
            self.wallet = dependencies.wallet
            self.network = dependencies.network
            self.contract_manager = dependencies.contract_manager
            self.database = dependencies.database
        else:
            self.deps = NodeDependencies(
                config_manager=None,
                rayonix_coin=None,
                wallet=None,
                network=None,
                contract_manager=None,
                database=None
            )
            self.config_manager = None
            self.rayonix_coin = None
            self.wallet = None
            self.network = None
            self.contract_manager = None
            self.database = None
        
        self.running = False
        self.shutdown_event = threading.Event()
        self.api_server = None
        self.background_tasks: Set[asyncio.Task] = set()
        
        # State management
        self.sync_state = {
            'syncing': False,
            'current_block': 0,
            'target_block': 0,
            'peers_connected': 0
        }
        
        # Command history
        self.command_history = []
        self.history_file = Path('.rayonix_history')
    
    async def initialize_components(self, config_path: Optional[str] = None, 
                                  encryption_key: Optional[str] = None) -> bool:
        """Initialize all node components with dependency injection support"""
        try:
            logger.info("Initializing RAYONIX Node components...")
            
            # Initialize config if not provided via dependencies
            if not self.config_manager:
                self.config_manager = self._initialize_config(config_path, encryption_key)
            
            self.config = self.config_manager.config
            
            # Initialize rayonix_coin if not provided via dependencies
            if not self.rayonix_coin:
                network_type = self.config.network.network_type
                data_dir = Path(self.config.database.db_path)
                data_dir.mkdir(exist_ok=True, parents=True)
                
                self.rayonix_coin = RayonixCoin(
                    network_type=network_type,
                    data_dir=str(data_dir)
                )
            
            # Initialize wallet if not provided via dependencies
            if not self.wallet:
                await self._initialize_wallet_with_blockchain()
            
            # Initialize network if enabled and not provided via dependencies
            if self.config.network.enabled and not self.network:
                await self._initialize_network()
            
            # Initialize API server if enabled
            if self.config.api.enabled:
                self.api_server = RayonixAPIServer(
                    self, 
                    self.config.api.host, 
                    self.config.api.port
                )
            
            logger.info("RAYONIX Node components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize node components: {e}")
            traceback.print_exc()
            return False
    
    def _initialize_config(self, config_path: Optional[str], encryption_key: Optional[str]) -> ConfigManager:
        """Initialize the configuration manager"""
        try:
            if config_path:
                config_manager = init_config(config_path, encryption_key, auto_reload=True)
            else:
                # Try default config paths
                default_paths = [
                    './rayonix.yaml',
                    './rayonix.yml', 
                    './rayonix.json',
                    './rayonix.toml',
                    './config/rayonix.yaml',
                    './config/rayonix.yml'
                ]
                
                for path in default_paths:
                    if Path(path).exists():
                        config_manager = init_config(path, encryption_key, auto_reload=True)
                        logger.info(f"Loaded config from {path}")
                        break
                else:
                    # No config file found, create default
                    config_manager = init_config(None, encryption_key, auto_reload=False)
                    logger.info("Using default configuration")
            
            return config_manager
            
        except Exception as e:
            logger.error(f"Failed to initialize config: {e}")
            return init_config(None, encryption_key, auto_reload=False)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        return self.config_manager.get(key, default)
    
    async def _initialize_wallet_with_blockchain(self):
        """Initialize wallet with proper blockchain reference integration"""
        wallet_file = Path(self.config.database.db_path) / 'wallet.dat'
        if wallet_file.exists():
            try:
                # Load existing wallet
                self.wallet = RayonixWallet()
                if self.wallet.restore(str(wallet_file)):
                    # Set blockchain reference after successful restore
                    if self.wallet.set_blockchain_reference(self.rayonix_coin):
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
            # Create new wallet
            self.wallet = create_new_wallet(
                wallet_type=WalletType.HD,
                network=self.config.network.network_type,
                address_type=AddressType.RAYONIX
            )
            # Establish blockchain reference
            if self.wallet.set_blockchain_reference(self.rayonix_coin):
                logger.info("New wallet created with blockchain integration")
                
                # Save wallet immediately
                wallet_file = Path(self.config.database.db_path) / 'wallet.dat'
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
    
    async def _initialize_network(self):
        """Initialize P2P network using config.py settings"""
        try:
            # Get network settings from config using attribute access
            network_config = NodeConfig(
                network_type=NetworkType[self.config.network.network_type.upper()],
                listen_ip=self.config.network.listen_ip,
                listen_port=self.config.network.listen_port,
                max_connections=self.config.network.max_connections,
                bootstrap_nodes=self.config.network.bootstrap_nodes,
                enable_encryption=self.config.network.enable_encryption,
                enable_compression=self.config.network.enable_compression,
                enable_dht=self.config.network.enable_dht,
                connection_timeout=self.config.network.connection_timeout,
                message_timeout=self.config.network.message_timeout
            )
            
            self.network = AdvancedP2PNetwork(network_config)
            
            # Register message handlers
            self.network.register_message_handler(
                MessageType.BLOCK, 
                self._handle_block_message
            )
            self.network.register_message_handler(
                MessageType.TRANSACTION,
                self._handle_transaction_message
            )
            
            logger.info("Network initialized with config settings")
            
        except Exception as e:
            logger.error(f"Failed to initialize network: {e}")
            self.network = None
            raise
    
    async def _handle_block_message(self, connection_id: str, message: Any):
        """Handle incoming block messages"""
        try:
            block_data = message.payload
            
            # RayonixCoin uses dict blocks, not Block objects
            if self.rayonix_coin._validate_block(block_data):
                self.rayonix_coin._add_block(block_data)
                logger.info(f"New block received: #{block_data['height']}")
                
                # Broadcast to other peers
                await self._broadcast_block(block_data)
                
        except Exception as e:
            logger.error(f"Error handling block: {e}")
    
    async def _handle_transaction_message(self, connection_id: str, message: Any):
        """Handle incoming transaction messages"""
        try:
            tx_data = message.payload
            
            # Add to mempool
            if self.rayonix_coin._validate_transaction(tx_data):
                self.rayonix_coin._add_to_mempool(tx_data)
                logger.info(f"New transaction received: {tx_data['hash'][:16]}...")
                
                # Broadcast to other peers
                await self._broadcast_transaction(tx_data)
                
        except Exception as e:
            logger.error(f"Error handling transaction: {e}")
    
    async def _broadcast_block(self, block: Dict):
        """Broadcast block to network"""
        if self.network:
            message = self.network.NetworkMessage(
                message_id=str(time.time()),
                message_type=self.network.MessageType.BLOCK,
                payload=block
            )
            await self.network.broadcast_message(message)
    
    async def _broadcast_transaction(self, transaction: Dict):
        """Broadcast transaction to network"""
        if self.network:
            message = self.network.NetworkMessage(
                message_id=str(time.time()),
                message_type=self.network.MessageType.TRANSACTION,
                payload=transaction
            )
            await self.network.broadcast_message(message)

    async def _staking_loop(self):
        """Background task for staking operations"""
        while self.running and not self.shutdown_event.is_set():
            try:
                if self.wallet and self.rayonix_coin:
                    # Check if we have enough balance to stake
                    balance_info = self.wallet.get_balance()
                    if balance_info.total >= self.config.consensus.min_stake:
                        # Get the first address from wallet
                        from_address = list(self.wallet.addresses.keys())[0]
                        # Try to stake a portion of available balance
                        stake_amount = min(balance_info.available // 2, self.config.consensus.max_stake)
                        if stake_amount >= self.config.consensus.min_stake:
                            try:
                                result = self.rayonix_coin.register_validator(stake_amount)
                                if result:
                                    logger.info(f"Staked {stake_amount} RXY for validation")
                            except Exception as stake_error:
                                logger.error(f"Error in staking: {stake_error}")
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Staking loop error: {e}")
                await asyncio.sleep(30)
    
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
            
            # Start network if enabled
            if self.network:
                self._create_background_task(self.network.start())
            
            # Start background tasks
            self._create_background_task(self._sync_blocks())
            self._create_background_task(self._monitor_peers())
            self._create_background_task(self._process_mempool())
            
            # Start staking if enabled
            if self.get_config_value('consensus.consensus_type', 'pos') == 'pos' and self.wallet:
                self._create_background_task(self._staking_loop())
            
            # Start API server if enabled
            if self.api_server:
                if not await self.api_server.start():
                    logger.error("Failed to start API server")
            
            logger.info("RAYONIX Node started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start node: {e}")
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
        
        # Save state through rayonix_coin
        if self.rayonix_coin:
            self.rayonix_coin.close()
        
        logger.info("RAYONIX Node stopped gracefully")
    
    async def _sync_blocks(self):
        """Synchronize blocks with network"""
        while self.running:
            try:
                if self.network and self.network.connections:
                    self.sync_state['syncing'] = True
                    await self._download_blocks()
                else:
                    self.sync_state['syncing'] = False
                
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Block sync error: {e}")
                await asyncio.sleep(30)
    
    async def _download_blocks(self):
        """Download missing blocks from peers"""
        current_height = len(self.rayonix_coin.blockchain)
        
        # Get highest block from peers
        highest_block = await self._get_highest_block()
        if highest_block > current_height:
            self.sync_state['target_block'] = highest_block
            self.sync_state['current_block'] = current_height
            
            logger.info(f"Syncing blocks {current_height} -> {highest_block}")
            
            # Download blocks in batches
            batch_size = 100
            for start_block in range(current_height, highest_block, batch_size):
                end_block = min(start_block + batch_size, highest_block)
                await self._download_block_batch(start_block, end_block)
                
                self.sync_state['current_block'] = end_block
                
                if not self.running:
                    break
    
    async def _get_highest_block(self) -> int:
        """Get highest block height from peers"""
        if not self.network or not self.network.connections:
            return len(self.rayonix_coin.blockchain)
        
        try:
            # Query all connected peers for their block height
            heights = []
            for peer_id, peer in self.network.connections.items():
                try:
                    # In a real implementation, this would send a message and await response
                    height = await self.network.query_peer_block_height(peer_id)
                    if height is not None:
                        heights.append(height)
                except:
                    continue
            
            if heights:
                return max(heights)
            else:
                return len(self.rayonix_coin.blockchain)
        except Exception as e:
            logger.error(f"Error getting highest block: {e}")
            return len(self.rayonix_coin.blockchain)
    
    async def _download_block_batch(self, start: int, end: int):
        """Download batch of blocks from peers"""
        if not self.network or not self.network.connections:
            return
        
        try:
            # Select a random peer to download from
            import random
            peer_id = random.choice(list(self.network.connections.keys()))
            
            # Request blocks from the peer
            blocks = await self.network.request_blocks(peer_id, start, end)
            
            if blocks:
                # Add blocks to our blockchain
                for block in blocks:
                    if self.rayonix_coin._validate_block(block):
                        self.rayonix_coin._add_block(block)
                        logger.debug(f"Added block #{block['height']} from peer {peer_id}")
            
        except Exception as e:
            logger.error(f"Error downloading block batch {start}-{end}: {e}")
    
    async def _monitor_peers(self):
        """Monitor peer connections and network health"""
        while self.running:
            try:
                if self.network:
                    self.sync_state['peers_connected'] = len(self.network.connections)
                
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Peer monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _process_mempool(self):
        """Process transactions in mempool"""
        while self.running:
            try:
                # RayonixCoin handles its own block creation internally
                # We just need to ensure network is connected
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Mempool processing error: {e}")
                await asyncio.sleep(10)
    
    def _save_state(self):
        """Save node state to disk"""
        try:
            # RayonixCoin handles its own state saving
            if hasattr(self.rayonix_coin, 'close'):
                self.rayonix_coin.close()
            
            logger.info("Node state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    async def handle_command(self, command: str, args: List[str]) -> bool:
        """Handle CLI commands with improved parsing and error handling"""
        try:
            # Parse command with better error handling
            if command == 'exit':
                await self.stop()
                return False
                
            elif command == 'help':
                self._show_help()
                
            elif command == 'status':
                self._show_status()
                
            elif command == 'create-wallet':
                self._create_wallet()
                
            elif command == 'load-wallet':
                if not args:
                    print("Error: Mnemonic phrase required")
                    print("Usage: load-wallet <mnemonic_phrase>")
                else:
                    self._load_wallet(args)
                
            elif command == 'get-balance':
                self._get_balance(args)
                
            elif command == 'send':
                if len(args) < 2:
                    print("Error: Recipient address and amount required")
                    print("Usage: send <to_address> <amount> [fee]")
                else:
                    self._send_transaction(args)
                
            elif command == 'stake':
                if not args:
                    print("Error: Amount required")
                    print("Usage: stake <amount>")
                else:
                    await self._stake_tokens(args)
                
            elif command == 'deploy-contract':
                if not args:
                    print("Error: Contract code required")
                    print("Usage: deploy-contract <contract_code> [initial_balance]")
                else:
                    await self._deploy_contract(args)
                
            elif command == 'call-contract':
                if len(args) < 2:
                    print("Error: Contract address and function required")
                    print("Usage: call-contract <address> <function> <args...>")
                else:
                    await self._call_contract(args)
                
            elif command == 'peers':
                self._show_peers()
                
            elif command == 'network-info':
                self._show_network_info()
                
            elif command == 'blockchain-info':
                self._show_blockchain_info()
                
            elif command == 'transaction':
                if not args:
                    print("Error: Transaction hash required")
                    print("Usage: transaction <hash>")
                else:
                    self._show_transaction(args)
                
            elif command == 'block':
                if not args:
                    print("Error: Block height or hash required")
                    print("Usage: block <height_or_hash>")
                else:
                    self._show_block(args)
                
            elif command == 'mempool':
                self._show_mempool()
                
            elif command == 'contracts':
                self._show_contracts()
                
            elif command == 'validator-info':
                self._show_validator_info()
                
            elif command == 'sync-status':
                self._show_sync_status()
                
            elif command == 'wallet-info':
                self._show_wallet_info()
                
            elif command == 'config':
                self._show_config_info(args)
                
            elif command == 'api-status':
                self._show_api_status()
                
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
            return True
            
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            print(f"Error: {e}")
            return True
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
  help                 - Show this help message
  exit                 - Stop the node and exit
  status               - Show node status
  config [key]         - Show configuration value(s)
  create-wallet        - Create a new wallet
  load-wallet <words>  - Load wallet from mnemonic phrase
  wallet-info          - Show loaded wallet information
  get-balance [addr]   - Get wallet or address balance
  send <to> <amount>   - Send coins to address
  stake <amount>       - Stake tokens for validation
  deploy-contract      - Deploy a smart contract
  call-contract        - Call a contract function
  peers                - Show connected peers
  network-info         - Show network information
  blockchain-info      - Show blockchain information
  transaction <hash>   - Show transaction details
  block <height/hash>  - Show block details
  mempool              - Show mempool transactions
  contracts            - List deployed contracts
  validator-info       - Show validator information
  sync-status          - Show synchronization status
  api-status           - Show API server status
"""
        print(help_text)
    
    def _show_status(self):
        """Show node status"""
        status = {
            'Running': self.running,
            'Network': self.get_config_value('network.network_type', 'mainnet'),
            'Network ID': self.get_config_value('network.network_id', 1),
            'Block Height': len(self.rayonix_coin.blockchain),
            'Connected Peers': self.sync_state['peers_connected'],
            'Syncing': self.sync_state['syncing'],
            'Consensus': self.get_config_value('consensus.consensus_type', 'pos'),
            'Wallet Loaded': self.wallet is not None,
            'API Enabled': self.get_config_value('api.enabled', False),
            'API Port': self.get_config_value('api.port', 8545)
        }
        
        for key, value in status.items():
            print(f"{key}: {value}")
    
    def _create_wallet(self):
        """Create a new wallet"""
        if self.wallet:
            print("Warning: Wallet already loaded. Create a new one? (y/N)")
            response = input().strip().lower()
            if response != 'y':
                return
        
        try:
            self.wallet = create_new_wallet(
                wallet_type=WalletType.HD,
                network=self.config.network.network_type,
                address_type=AddressType.RAYONIX
            )
            
            # Set blockchain reference
            if self.wallet.set_blockchain_reference(self.rayonix_coin):
                # Save wallet
                wallet_file = Path(self.get_config_value('database.db_path', './rayonix_data')) / 'wallet.dat'
                if self.wallet.backup(str(wallet_file)):
                    print("New wallet created and saved successfully")
                    print(f"Mnemonic: {self.wallet.get_mnemonic()}")
                    print(f"Address: {list(self.wallet.addresses.keys())[0]}")
                else:
                    print("Error: Failed to save wallet")
            else:
                print("Error: Failed to establish blockchain integration")
                self.wallet = None
                
        except Exception as e:
            print(f"Error creating wallet: {e}")
    
    def _load_wallet(self, args: List[str]):
        """Load wallet from mnemonic phrase"""
        if self.wallet:
            print("Warning: Wallet already loaded. Replace it? (y/N)")
            response = input().strip().lower()
            if response != 'y':
                return
        
        try:
            mnemonic = ' '.join(args)
            self.wallet = RayonixWallet()
            
            if self.wallet.restore_from_mnemonic(mnemonic):
                # Set blockchain reference
                if self.wallet.set_blockchain_reference(self.rayonix_coin):
                    # Save wallet
                    wallet_file = Path(self.get_config_value('database.db_path', './rayonix_data')) / 'wallet.dat'
                    if self.wallet.backup(str(wallet_file)):
                        print("Wallet loaded successfully")
                        print(f"Address: {list(self.wallet.addresses.keys())[0]}")
                    else:
                        print("Error: Failed to save wallet")
                else:
                    print("Error: Failed to establish blockchain integration")
                    self.wallet = None
            else:
                print("Error: Invalid mnemonic phrase")
                
        except Exception as e:
            print(f"Error loading wallet: {e}")
    
    def _get_balance(self, args: List[str]):
        """Get balance for address or wallet"""
        try:
            if args:
                # Get balance for specific address
                address = args[0]
                if not validate_rayonix_address(address):
                    print("Error: Invalid address format")
                    return
                
                balance = self.rayonix_coin.get_balance(address)
                print(f"Balance for {address}: {balance} RXY")
            else:
                # Get wallet balance
                if not self.wallet:
                    print("Error: No wallet loaded")
                    return
                
                balance_info = self.wallet.get_balance()
                print(f"Wallet Balance: {balance_info.total} RXY")
                print(f"Available: {balance_info.available} RXY")
                print(f"Pending: {balance_info.pending} RXY")
                print(f"Staked: {balance_info.staked} RXY")
                
        except Exception as e:
            print(f"Error getting balance: {e}")
    
    def _send_transaction(self, args: List[str]):
        """Send transaction to address"""
        try:
            if not self.wallet:
                print("Error: No wallet loaded")
                return
            
            to_address = args[0]
            amount = float(args[1])
            fee = float(args[2]) if len(args) > 2 else 0.001
            
            if not validate_rayonix_address(to_address):
                print("Error: Invalid recipient address")
                return
            
            # Get the first address from wallet
            from_address = list(self.wallet.addresses.keys())[0]
            
            # Create transaction
            transaction = self.rayonix_coin.create_transaction(
                from_address, to_address, amount, fee
            )
            
            if not transaction:
                print("Error: Failed to create transaction")
                return
            
            # Sign transaction
            signed_tx = self.wallet.sign_transaction(transaction)
            
            if not signed_tx:
                print("Error: Failed to sign transaction")
                return
            
            # Add to mempool
            if self.rayonix_coin._validate_transaction(signed_tx):
                self.rayonix_coin._add_to_mempool(signed_tx)
                print(f"Transaction created: {signed_tx['hash']}")
                
                # Broadcast to network if available
                if self.network:
                    asyncio.create_task(self._broadcast_transaction(signed_tx))
            else:
                print("Error: Invalid transaction")
                
        except Exception as e:
            print(f"Error sending transaction: {e}")
    
    async def _stake_tokens(self, args: List[str]):
        """Stake tokens for validation"""
        try:
            if not self.wallet:
                print("Error: No wallet loaded")
                return
            
            amount = float(args[0])
            
            if amount < self.get_config_value('consensus.min_stake', 1000):
                print(f"Error: Minimum stake is {self.get_config_value('consensus.min_stake', 1000)} RXY")
                return
            
            # Get the first address from wallet
            from_address = list(self.wallet.addresses.keys())[0]
            
            # Register as validator
            result = self.rayonix_coin.register_validator(amount)
            
            if result:
                print(f"Successfully staked {amount} RXY for validation")
            else:
                print("Error: Failed to stake tokens")
                
        except Exception as e:
            print(f"Error staking tokens: {e}")
    
    async def _deploy_contract(self, args: List[str]):
        """Deploy a smart contract"""
        try:
            if not self.wallet:
                print("Error: No wallet loaded")
                return
            
            if not self.contract_manager:
                print("Error: Contract manager not available")
                return
            
            contract_code = args[0]
            initial_balance = float(args[1]) if len(args) > 1 else 0.0
            
            # Get the first address from wallet
            from_address = list(self.wallet.addresses.keys())[0]
            
            # Deploy contract
            contract_address = await self.contract_manager.deploy_contract(
                from_address, contract_code, initial_balance
            )
            
            if contract_address:
                print(f"Contract deployed at: {contract_address}")
            else:
                print("Error: Failed to deploy contract")
                
        except Exception as e:
            print(f"Error deploying contract: {e}")
    
    async def _call_contract(self, args: List[str]):
        """Call a contract function"""
        try:
            if not self.wallet:
                print("Error: No wallet loaded")
                return
            
            if not self.contract_manager:
                print("Error: Contract manager not available")
                return
            
            contract_address = args[0]
            function_name = args[1]
            function_args = args[2:] if len(args) > 2 else []
            
            # Get the first address from wallet
            from_address = list(self.wallet.addresses.keys())[0]
            
            # Call contract
            result = await self.contract_manager.call_contract(
                from_address, contract_address, function_name, function_args
            )
            
            if result:
                print(f"Contract call result: {result}")
            else:
                print("Error: Failed to call contract")
                
        except Exception as e:
            print(f"Error calling contract: {e}")
    
    def _show_peers(self):
        """Show connected peers"""
        if not self.network:
            print("Network not enabled")
            return
        
        peers = self.network.get_connected_peers()
        if not peers:
            print("No peers connected")
            return
        
        print(f"Connected Peers ({len(peers)}):")
        for peer_id, peer_info in peers.items():
            print(f"  {peer_id}: {peer_info.get('address', 'Unknown')}")
    
    def _show_network_info(self):
        """Show network information"""
        if not self.network:
            print("Network not enabled")
            return
        
        metrics = self.network.get_metrics() or {}
        
        print("Network Information:")
        print(f"  Connections: {metrics.get('connections', 0)}")
        print(f"  Messages Sent: {metrics.get('messages_sent', 0)}")
        print(f"  Messages Received: {metrics.get('messages_received', 0)}")
        print(f"  Bytes Sent: {metrics.get('bytes_sent', 0)}")
        print(f"  Bytes Received: {metrics.get('bytes_received', 0)}")
        print(f"  Uptime: {metrics.get('uptime', 0)} seconds")
    
    def _show_blockchain_info(self):
        """Show blockchain information"""
        blockchain = self.rayonix_coin.blockchain
        
        print("Blockchain Information:")
        print(f"  Height: {len(blockchain)}")
        
        if blockchain:
            last_block = blockchain[-1]
            print(f"  Latest Block: #{last_block.get('height', 0)}")
            print(f"  Hash: {last_block.get('hash', 'Unknown')}")
            print(f"  Timestamp: {datetime.fromtimestamp(last_block.get('timestamp', 0))}")
            print(f"  Transactions: {len(last_block.get('transactions', []))}")
        
        print(f"  Mempool Size: {len(self.rayonix_coin.mempool)}")
    
    def _show_transaction(self, args: List[str]):
        """Show transaction details"""
        tx_hash = args[0]
        transaction = self.rayonix_coin.get_transaction(tx_hash)
        
        if not transaction:
            print("Transaction not found")
            return
        
        print("Transaction Details:")
        print(f"  Hash: {transaction.get('hash')}")
        print(f"  Block: {transaction.get('block_height', 'Unconfirmed')}")
        print(f"  Timestamp: {datetime.fromtimestamp(transaction.get('timestamp', 0))}")
        print(f"  Inputs: {len(transaction.get('inputs', []))}")
        print(f"  Outputs: {len(transaction.get('outputs', []))}")
        
        # Show inputs
        print("  Inputs:")
        for i, tx_input in enumerate(transaction.get('inputs', [])):
            print(f"    {i}: {tx_input.get('address', 'Unknown')} - {tx_input.get('amount', 0)} RXY")
        
        # Show outputs
        print("  Outputs:")
        for i, tx_output in enumerate(transaction.get('outputs', [])):
            print(f"    {i}: {tx_output.get('address', 'Unknown')} - {tx_output.get('amount', 0)} RXY")
    
    def _show_block(self, args: List[str]):
        """Show block details"""
        identifier = args[0]
        
        try:
            # Try to parse as height first
            try:
                height = int(identifier)
                block = self.rayonix_coin.get_block(height)
            except ValueError:
                # Treat as hash
                block = self.rayonix_coin.get_block(identifier)
            
            if not block:
                print("Block not found")
                return
            
            print("Block Details:")
            print(f"  Height: {block.get('height', 'Unknown')}")
            print(f"  Hash: {block.get('hash', 'Unknown')}")
            print(f"  Previous Hash: {block.get('previous_hash', 'Genesis')}")
            print(f"  Timestamp: {datetime.fromtimestamp(block.get('timestamp', 0))}")
            print(f"  Transactions: {len(block.get('transactions', []))}")
            print(f"  Validator: {block.get('validator', 'Unknown')}")
            print(f"  Signature: {block.get('signature', 'Unknown')[:16]}...")
            
        except Exception as e:
            print(f"Error getting block: {e}")
    
    def _show_mempool(self):
        """Show mempool transactions"""
        mempool = self.rayonix_coin.mempool
        
        print(f"Mempool ({len(mempool)} transactions):")
        for i, tx in enumerate(mempool[:10]):  # Show first 10
            print(f"  {i+1}: {tx.get('hash', 'Unknown')[:16]}... - {sum(out.get('amount', 0) for out in tx.get('outputs', []))} RXY")
        
        if len(mempool) > 10:
            print(f"  ... and {len(mempool) - 10} more")
    
    def _show_contracts(self):
        """Show deployed contracts"""
        if not self.contract_manager:
            print("Contract manager not available")
            return
        
        contracts = self.contract_manager.get_deployed_contracts()
        
        if not contracts:
            print("No contracts deployed")
            return
        
        print("Deployed Contracts:")
        for address, contract in contracts.items():
            print(f"  {address}: {contract.get('name', 'Unknown')}")
    
    def _show_validator_info(self):
        """Show validator information"""
        validators = self.rayonix_coin.get_validators()
        
        if not validators:
            print("No validators found")
            return
        
        print("Validators:")
        for address, stake in validators.items():
            print(f"  {address}: {stake} RXY staked")
    
    def _show_sync_status(self):
        """Show synchronization status"""
        print("Synchronization Status:")
        print(f"  Syncing: {self.sync_state['syncing']}")
        print(f"  Current Block: {self.sync_state['current_block']}")
        print(f"  Target Block: {self.sync_state['target_block']}")
        print(f"  Connected Peers: {self.sync_state['peers_connected']}")
    
    def _show_wallet_info(self):
        """Show wallet information"""
        if not self.wallet:
            print("No wallet loaded")
            return
        
        addresses = self.wallet.addresses
        balance_info = self.wallet.get_balance()
        
        print("Wallet Information:")
        print(f"  Type: {self.wallet.wallet_type}")
        print(f"  Address Count: {len(addresses)}")
        print(f"  Total Balance: {balance_info.total} RXY")
        print(f"  Available: {balance_info.available} RXY")
        print(f"  Pending: {balance_info.pending} RXY")
        print(f"  Staked: {balance_info.staked} RXY")
        
        print("  Addresses:")
        for i, (address, info) in enumerate(list(addresses.items())[:5]):  # Show first 5
            print(f"    {i+1}: {address} - {info.get('balance', 0)} RXY")
        
        if len(addresses) > 5:
            print(f"    ... and {len(addresses) - 5} more")
    
    def _show_config_info(self, args: List[str]):
        """Show configuration information"""
        if args:
            # Show specific config key
            key = args[0]
            value = self.get_config_value(key, "Not found")
            print(f"{key}: {value}")
        else:
            # Show all config
            config_dict = self.config_manager.get_all()
            print("Configuration:")
            for section, settings in config_dict.items():
                print(f"  [{section}]")
                for key, value in settings.items():
                    print(f"    {key}: {value}")
    
    def _show_api_status(self):
        """Show API server status"""
        if not self.api_server:
            print("API server not enabled")
            return
        
        print("API Server Status:")
        print(f"  Running: {self.running}")
        print(f"  Host: {self.get_config_value('api.host', '127.0.0.1')}")
        print(f"  Port: {self.get_config_value('api.port', 8545)}")
        print(f"  JSON-RPC: Enabled")
        print(f"  REST API: Enabled")

    def _load_command_history(self):
        """Load command history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    for line in f:
                        self.command_history.append(line.strip())
        except Exception as e:
            logger.error(f"Error loading command history: {e}")
    
    def _save_command_history(self):
        """Save command history to file"""
        try:
            with open(self.history_file, 'w') as f:
                for command in self.command_history[-100:]:  # Keep last 100 commands
                    f.write(command + '\n')
        except Exception as e:
            logger.error(f"Error saving command history: {e}")

async def main():
    """Main function with enhanced CLI using argparse"""
    parser = argparse.ArgumentParser(description='RAYONIX Blockchain Node')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--network', '-n', choices=['mainnet', 'testnet', 'regtest'], 
                       help='Network type')
    parser.add_argument('--port', '-p', type=int, help='P2P network port')
    parser.add_argument('--api-port', type=int, help='API server port')
    parser.add_argument('--no-api', action='store_true', help='Disable API server')
    parser.add_argument('--no-network', action='store_true', help='Disable P2P network')
    parser.add_argument('--data-dir', help='Data directory path')
    parser.add_argument('--encryption-key', help='Configuration encryption key')
    parser.add_argument('--wallet-mnemonic', help='Wallet mnemonic phrase')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Start in interactive mode (default)')
    parser.add_argument('--daemon', '-d', action='store_true', 
                       help='Start as daemon (non-interactive)')
    
    args = parser.parse_args()
    
    # Create node instance with dependency injection
    node = RayonixNode()
    
    # Initialize components
    if not await node.initialize_components(args.config, args.encryption_key):
        logger.error("Failed to initialize node components")
        return 1
    
    # Override config with command line arguments if provided
    if args.network:
        node.config_manager.set('network.network_type', args.network)
    
    if args.port:
        node.config_manager.set('network.listen_port', args.port)
    
    if args.api_port:
        node.config_manager.set('api.port', args.api_port)
    
    if args.no_api:
        node.config_manager.set('api.enabled', False)
    
    if args.no_network:
        node.config_manager.set('network.enabled', False)
    
    if args.data_dir:
        node.config_manager.set('database.db_path', args.data_dir)
    
    # Load wallet from mnemonic if provided
    if args.wallet_mnemonic and node.wallet is None:
        try:
            node.wallet = RayonixWallet()
            if node.wallet.restore_from_mnemonic(args.wallet_mnemonic):
                logger.info("Wallet loaded from command line mnemonic")
            else:
                logger.error("Failed to load wallet from mnemonic")
        except Exception as e:
            logger.error(f"Error loading wallet from mnemonic: {e}")
    
    # Start the node
    if not await node.start():
        logger.error("Failed to start node")
        return 1
    
    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(node.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run in appropriate mode
    if args.daemon:
        logger.info("Running in daemon mode")
        try:
            # Keep the node running until shutdown signal
            while node.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
    else:
        # Interactive mode with enhanced CLI
        logger.info("Running in interactive mode")
        print("RAYONIX Blockchain Node - Type 'help' for commands")
        
        # Load command history
        node._load_command_history()
        if node.command_history:
            for cmd in node.command_history:
                readline.add_history(cmd)
        
        try:
            while node.running:
                try:
                    # Get user input with readline support
                    command_input = input("rayonix> ").strip()
                    
                    if not command_input:
                        continue
                    
                    # Add to history
                    readline.add_history(command_input)
                    node.command_history.append(command_input)
                    
                    # Parse command
                    parts = command_input.split()
                    command = parts[0].lower()
                    cmd_args = parts[1:] if len(parts) > 1 else []
                    
                    # Handle command
                    if not await node.handle_command(command, cmd_args):
                        break
                        
                except EOFError:
                    print()  # New line after Ctrl+D
                    break
                except KeyboardInterrupt:
                    print()  # New line after Ctrl+C
                    print("Type 'exit' to quit or 'help' for commands")
                except Exception as e:
                    logger.error(f"Error in command loop: {e}")
                    print(f"Error: {e}")
        
        finally:
            # Save command history
            node._save_command_history()
    
    # Ensure node is stopped
    if node.running:
        await node.stop()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)