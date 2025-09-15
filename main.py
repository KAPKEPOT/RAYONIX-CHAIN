# main.py
import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import signal
import threading
from dataclasses import asdict
import readline

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

class RayonixNode:
    """Complete RAYONIX blockchain node using rayonix_coin.py as backend"""
    
    def __init__(self, config_path: Optional[str] = None, encryption_key: Optional[str] = None):
        self.config_manager = self._initialize_config(config_path, encryption_key)
        self.config = self.config_manager.config
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Initialize components
        self.rayonix_coin = None
        self.wallet = None
        self.network = None
        self.contract_manager = None
        self.database = None        
        
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
    
    async def initialize(self):
        """Initialize all node components using config.py settings"""
        try:
            logger.info("Initializing RAYONIX Node...")
            
            # Get configuration values from config.py
            network_type = self.config.network.network_type
            data_dir = Path(self.config.database.db_path)
            
            # Create data directory
            data_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize rayonix_coin with proper attribute access
            self.rayonix_coin = RayonixCoin(
                network_type=network_type,
                data_dir=str(data_dir)
            )
          
            # Initialize wallet with proper blockchain integration
            await self._initialize_wallet_with_blockchain()
            
            # Initialize network if enabled
            if self.config.network.enabled:
            	
            	await self._initialize_network()
            logger.info("RAYONIX Node initialized successfully with blockchain-wallet integration")
            return True
        
  
        except Exception as e:
            logger.error(f"Failed to initialize node: {e}")
            traceback.print_exc()
            return False
            
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
                asyncio.create_task(self.network.start())
            
            # Start background tasks
            asyncio.create_task(self._sync_blocks())
            asyncio.create_task(self._monitor_peers())
            asyncio.create_task(self._process_mempool())
            
            # Start staking if enabled
            if self.get_config_value('consensus.consensus_type', 'pos') == 'pos' and self.wallet:
                asyncio.create_task(self._staking_loop())
            
            logger.info("RAYONIX Node started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start node: {e}")
            traceback.print_exc()
            return False
    
    async def stop(self):
        """Stop the node gracefully"""
        if not self.running:
            return
        
        logger.info("Stopping RAYONIX Node...")
        self.running = False
        self.shutdown_event.set()
        
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
        # Implementation would query multiple peers
        return len(self.rayonix_coin.blockchain)  # Placeholder
    
    async def _download_block_batch(self, start: int, end: int):
        """Download batch of blocks"""
        # Implementation would request blocks from peers
        pass
    
    async def _monitor_peers(self):
        """Monitor peer connections and network health"""
        while self.running:
            try:
                if self.network:
                    self.sync_state['peers_connected'] = len(self.network.connections)
                
                await asyncio.sleep(30)
                
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
        """Handle CLI commands"""
        try:
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
                self._load_wallet(args)
                
            elif command == 'get-balance':
                self._get_balance(args)
                
            elif command == 'send':
                self._send_transaction(args)
                
            elif command == 'stake':
                await self._stake_tokens(args)
                
            elif command == 'deploy-contract':
                await self._deploy_contract(args)
                
            elif command == 'call-contract':
                await self._call_contract(args)
                
            elif command == 'peers':
                self._show_peers()
                
            elif command == 'network-info':
                self._show_network_info()
                
            elif command == 'blockchain-info':
                self._show_blockchain_info()
                
            elif command == 'transaction':
                self._show_transaction(args)
                
            elif command == 'block':
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
                
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
            return True
            
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            traceback.print_exc()
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
            'Block Time': self.get_config_value('consensus.block_time', 30),
            'Staking Enabled': self.get_config_value('consensus.consensus_type', 'pos') == 'pos',
            'Wallet Loaded': self.wallet is not None,
            'API Enabled': self.get_config_value('api.enabled', False),
            'API Port': self.get_config_value('api.port', 8545),
            'Log Level': self.get_config_value('logging.level', 'INFO')
        }
        
        print("Node Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    
    def _show_config_info(self, args: List[str]):
        """Show configuration values"""
        if not args:
            # Show all config sections
            print("Configuration Sections:")
            print("  network     - Network settings")
            print("  consensus   - Consensus parameters")
            print("  database    - Database configuration")
            print("  wallet      - Wallet settings")
            print("  api         - API configuration")
            print("  logging     - Logging settings")
            print("  security    - Security parameters")
            print("  Use 'config <section>' for details")
        else:
            key = args[0]
            value = self.get_config_value(key)
            print(f"Config {key}: {value}")
    
    def _create_wallet(self):
        """Create a new wallet"""
        try:
            # Create new wallet which generates a mnemonic
            wallet = create_new_wallet()
            
            # Get the mnemonic from the wallet
            mnemonic_phrase = wallet.get_mnemonic()
            
            if mnemonic_phrase:
                self.wallet = wallet
                print("✓ New wallet created successfully!")
                print(f"  Mnemonic: {mnemonic_phrase}")
                print("  🔐 IMPORTANT: Save this mnemonic phrase securely!")
                print("  🔐 You will need it to restore your wallet!")
                
                # Show the first address
                addresses = wallet.get_addresses()
                if addresses:
                    print(f"  Address: {addresses[0]}")
                else:
                    # Generate first address
                    address_info = wallet.derive_address(0, False)
                    print(f"  Address: {address_info.address}")
            else:
                print("✗ Failed to create wallet - no mnemonic generated")		
                
        except Exception as e:
            print(f"✗ Error creating wallet: {e}")
            traceback.print_exc()            
 
    def _load_wallet(self, args: List[str]):
        """Load wallet from file"""
        if not args:
            print("Usage: load-wallet <mnemonic_phrase>")
            return
            
        mnemonic_phrase = ' '.join(args)
        try:
            # Create wallet with default config
            wallet = RayonixWallet()
            
            if wallet.create_from_mnemonic(mnemonic_phrase):
                self.wallet = wallet
                print("✓ Wallet loaded successfully from mnemonic!")
                
                addresses = wallet.get_addresses()
                if addresses:
                    print(f"  Addresses loaded: {len(addresses)}")
                    print(f"  Primary address: {addresses[0]}")
                    
                    try:
                        balance_info = wallet.get_balance()
                        print(f"  Balance: {balance_info.total} RXY")
                    except:
                        print("  Balance: Unable to check (wallet not synced)")
                else:
                    print("  No addresses found in wallet")
            else:
                print("  No addresses found in wallet")
        except Exception as e:
            print(f"✗ Error loading wallet: {e}")
            print("Make sure the mnemonic phrase is correct (usually 12, 18, or 24 words)")
            
    def _show_wallet_info(self):
        """Show information about the currently loaded wallet"""
        if not self.wallet:
            print("No wallet loaded. Use 'create-wallet' or 'load-wallet'")
            return
        
        print("Wallet Information:")
        print(f"  Loaded: Yes")
        
        if hasattr(self.wallet, 'addresses') and self.wallet.addresses:
            addresses = list(self.wallet.addresses.keys())
            print(f"  Addresses: {len(addresses)}")
            print(f"  Primary: {addresses[0]}")
            
            # Get balance with error handling
            try:
            	balance_info = self.wallet.get_balance()
            	
            	# Show balances
            	total_balance = balance_info.total
            	print(f"  Total Balance: {total_balance} RXY")
            	
            	if hasattr(balance_info, 'offline_mode') and balance_info.offline_mode:
            		print(f"  Mode: Offline")
            		if hasattr(balance_info, 'last_online_update') and balance_info.last_online_update:
            			print(f"  Last Update: {datetime.fromtimestamp(balance_info.last_online_update).strftime('%Y-%m-%d %H:%M:%S')}")
            			if hasattr(balance_info, 'confidence_level') and balance_info.confidence_level:
            				print(f"  Confidence: {balance_info.confidence_level}")
            				if balance_info.confidence_level == "low":
            					print("  ⚠️  Warning: Balance may be inaccurate - connect to network")
            			else:
            				print(f"  Mode: Online")
            				print(f"  Confirmed: {balance_info.confirmed} RXY")
            				print(f"  Unconfirmed: {balance_info.unconfirmed} RXY")
            				print(f"  Available: {balance_info.available} RXY")
            				print(f"  Locked: {balance_info.locked} RXY")
            			
            			# Show individual address balances
            			if hasattr(balance_info, 'by_address') and balance_info.by_address:
            				print("  Address Balances:")
            				for address, addr_balance in list(balance_info.by_address.items())[:5]:
            					if isinstance(addr_balance, dict):
            						balance_val = addr_balance.get('total', 0)
            						offline_flag = " (offline)" if addr_balance.get('offline_estimated') else ""
            						print(f"    {address}: {balance_val} RXY{offline_flag}")
            					else:
            					    print(f"    {address}: {addr_balance} RXY")
            		if hasattr(self.wallet, 'get_master_xpub'):
            				xpub = self.wallet.get_master_xpub()
            				if xpub:
            					print(f"  Master xpub: {xpub[:30]}...")
            except Exception as e:
                print(f"  Error getting balance: {e}")
                print("  Balance information unavailable")				       
            				
                        
                        
                        
            
            		
            
        
            
        
    def _get_balance(self, args: List[str]):
        """Get balance for address or loaded wallet"""
        try:
            if args:
                # Specific address requested
                address = args[0]
                balance = self.rayonix_coin.get_balance(address)
                print(f"Balance for {address}: {balance} RXY")
            elif self.wallet:
                # Use wallet's get_balance method which handles offline mode
                balance_info = self.wallet.get_balance()
                
                # Check if we're in offline mode
                if balance_info.offline_mode:
                	print(f"Offline Mode: {balance_info.total} RXY")
                	if balance_info.last_online_update:
                		print(f"Last updated: {datetime.fromtimestamp(balance_info.last_online_update).strftime('%Y-%m-%d %H:%M:%S')}")
                	if balance_info.confidence_level:
                		print(f"Confidence: {balance_info.confidence_level}")
                		if balance_info.confidence_level == "low":
                			print("Warning: Balance may be inaccurate - connect to network")
                	if balance_info.warning:
                		print(f"Note: {balance_info.warning}")
                else:
                	print(f"Online Balance: {balance_info.total} RXY")
                	print(f"  Confirmed: {balance_info.confirmed} RXY")
                	print(f"  Unconfirmed: {balance_info.unconfirmed} RXY")
                	print(f"  Available: {balance_info.available} RXY")
                	print(f"  Locked: {balance_info.locked} RXY")           
            else:
                print("No address specified and no wallet loaded")
        except Exception as e:
                print(f"Error getting balance: {e}")
                
                
    def _send_transaction(self, args: List[str]):
        """Send transaction"""
        if not self.wallet or not self.wallet.addresses:
            print("No wallet loaded or no addresses in wallet")
            return
        
        if len(args) < 2:
            print("Usage: send <to_address> <amount> [fee]")
            return
        
        to_address = args[0]
        amount = int(args[1])
        fee = int(args[2]) if len(args) > 2 else 1

        # Get the first address from wallet as sender
        from_address = list(self.wallet.addresses.keys())[0]
        
        # First check balance
        try:
        	balance = self.rayonix_coin.get_balance(from_address)
        	if balance < amount + fee:
        		print(f"Error: Insufficient funds. Balance: {balance} RXY, Required: {amount + fee} RXY")
        		return
        except Exception as e:
        	print(f"Error checking balance: {e}")
        	return
        
        # Create and send transaction using rayonix_coin
        try:
            transaction = self.rayonix_coin.create_transaction(
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                fee=fee
            )
            if transaction is None:
            	print("Transaction creation failed: No spendable funds or validation error")
            	return
            
            print(f"Transaction sent successfully!")
            print(f"TXID: {transaction['hash'][:16]}...")
            
        except Exception as e:
            print(f"Error sending transaction: {e}")
    
    async def _stake_tokens(self, args: List[str]):
        """Stake tokens for validation"""
        if not self.wallet:
            print("No wallet loaded")
            return
        
        amount = int(args[0]) if args else self.rayonix_coin.get_balance(self.wallet.address)
        
        try:
            result = self.rayonix_coin.register_validator(amount)
            if result:
                print(f"Staked {amount} RXY for validation")
            else:
                print("Failed to stake tokens")
        except Exception as e:
            print(f"Error staking tokens: {e}")
    
    async def _deploy_contract(self, args: List[str]):
        """Deploy smart contract"""
        if not self.wallet:
            print("No wallet loaded")
            return
        
        if len(args) < 1:
            print("Usage: deploy-contract <contract_code> [initial_balance]")
            return
        
        contract_code = args[0]
        initial_balance = int(args[1]) if len(args) > 1 else 0
        
        try:
            contract_address = self.rayonix_coin.deploy_contract(
                contract_code, 
                initial_balance
            )
            print(f"Contract deployed at: {contract_address}")
        except Exception as e:
            print(f"Error deploying contract: {e}")
    
    async def _call_contract(self, args: List[str]):
        """Call contract function"""
        if not self.wallet:
            print("No wallet loaded")
            return
        
        if len(args) < 3:
            print("Usage: call-contract <address> <function> <args...>")
            return
        
        contract_address, function_name = args[0], args[1]
        function_args = args[2:]
        
        try:
            result = self.rayonix_coin.call_contract(
                contract_address,
                function_name,
                function_args
            )
            print(f"Contract call result: {result}")
        except Exception as e:
            print(f"Error calling contract: {e}")
    
    def _show_peers(self):
        """Show connected peers"""
        if not self.network:
            print("Network not enabled")
            return
        try:
            peers = self.network.get_connected_peers()
            if peers is None:
                print("Peer information unavailable")
                return
            print(f"Connected Peers ({len(peers)}):")
            for peer in peers:
                print(f"  {peer}")
        except Exception as e:
            print(f"Error displaying peers: {e}")
    
    def _show_network_info(self):
        """Show network information"""
        if not self.network:
            print("Network not enabled")
            return
        try:
            info = self.network.get_metrics()
            
            # Ensure info is not None
            if info is None:
                print("Network information unavailable (metrics returned None)")
                return
            print("Network Information:")
            print(f"  Node ID: {info.get('node_id', 'N/A')}")
            print(f"  Network Type: {info.get('network_type', 'N/A')}")
            print(f"  Listen Address: {info.get('listen_address', 'N/A')}")
            print(f"  Active Connections: {info.get('active_connections', 0)}")
            print(f"  Known Peers: {info.get('known_peers', 0)}")
            print(f"  Total Bytes Sent: {info.get('total_bytes_sent', 0)}")
            print(f"  Total Bytes Received: {info.get('total_bytes_received', 0)}")
            print(f"  Encryption: {'Enabled' if info.get('encryption_enabled') else 'Disabled'}")
            print(f"  Compression: {'Enabled' if info.get('compression_enabled') else 'Disabled'}")
            
            # Show connection details if available
            connections = info.get('connections', {})
            if connections:
                print(f"  Connection Details ({len(connections)}):")
                for conn_id, conn_info in connections.items():
                    print(f"    {conn_id}:")
                    print(f"      Protocol: {conn_info.get('protocol', 'N/A')}")
                    print(f"      Bytes Sent: {conn_info.get('bytes_sent', 0)}")
                    print(f"      Bytes Received: {conn_info.get('bytes_received', 0)}")
                    print(f"      Last Activity: {conn_info.get('last_activity', 0):.1f}s ago")
            # Show error if present
            if 'error' in info:
                print(f"  Error: {info['error']}") 
              
        except Exception as e:
            print(f"Error displaying network information: {e}")
            traceback.print_exc()        
    
    def _show_blockchain_info(self):
        """Show blockchain information"""
        info = self.rayonix_coin.get_blockchain_info()
        
        print("Blockchain Information:")
        print(f"  Network: {self.get_config_value('network.network_type', 'mainnet')}")
        print(f"  Height: {info.get('height', 0)}")
        print(f"  Consensus: {self.get_config_value('consensus.consensus_type', 'pos')}")
        print(f"  Block Time: {self.get_config_value('consensus.block_time', 30)}s")
        print(f"  Block Reward: {self.get_config_value('consensus.block_reward', 50)} RXY")
        print(f"  Min Stake: {self.get_config_value('consensus.min_stake', 1000)} RXY")
        print(f"  Total Supply: {info.get('total_supply', 0)} RXY")
        print(f"  Mempool Size: {len(self.rayonix_coin.mempool)}")
    
    def _show_transaction(self, args: List[str]):
        """Show transaction details"""
        if len(args) < 1:
            print("Usage: transaction <hash>")
            return
        
        tx_hash = args[0]
        transaction = self.rayonix_coin.get_transaction(tx_hash)
        
        if transaction:
            print(f"Transaction {tx_hash}:")
            print(f"  From: {transaction.get('from', 'Unknown')}")
            print(f"  To: {transaction.get('to', 'Unknown')}")
            print(f"  Amount: {transaction.get('amount', 0)} RXY")
            print(f"  Fee: {transaction.get('fee', 0)} RXY")
            print(f"  Block: {transaction.get('block_height', 'Pending')}")
        else:
            print(f"Transaction {tx_hash} not found")
    
    def _show_block(self, args: List[str]):
        """Show block details"""
        if len(args) < 1:
            print("Usage: block <height_or_hash>")
            return
        
        identifier = args[0]
        block = None
        
        # Try to get by height first
        try:
            height = int(identifier)
            block = self.rayonix_coin.get_block(height)
        except ValueError:
            # Try to get by hash
            block = self.rayonix_coin.get_block(identifier)
        
        if block:
            print(f"Block #{block.get('height', 'Unknown')}:")
            print(f"  Hash: {block.get('hash', 'Unknown')}")
            print(f"  Previous: {block.get('previous_hash', 'Unknown')}")
            print(f"  Validator: {block.get('validator', 'Unknown')}")
            print(f"  Transactions: {len(block.get('transactions', []))}")
            print(f"  Timestamp: {time.ctime(block.get('timestamp', 0))}")
        else:
            print(f"Block {identifier} not found")
    
    def _show_mempool(self):
        """Show mempool transactions"""
        mempool = self.rayonix_coin.mempool
        print(f"Mempool Transactions ({len(mempool)}):")
        
        for tx in mempool[:10]:  # Show first 10
            tx_hash = tx.get('hash', 'Unknown')[:16] + '...'
            from_addr = tx.get('from', 'Unknown')[:10] + '...'
            to_addr = tx.get('to', 'Unknown')[:10] + '...'
            amount = tx.get('amount', 0)
            print(f"  {tx_hash}: {from_addr} -> {to_addr} {amount} RXY")
    
    def _show_contracts(self):
        """Show deployed contracts"""
        # This would need to be implemented in rayonix_coin
        print("Contract listing not yet implemented in rayonix_coin")
    
    def _show_validator_info(self):
        """Show validator information"""
        if not self.wallet:
            print("No wallet loaded")
            return
        
        # This would need validator info methods in rayonix_coin
        print("Validator info not yet implemented in rayonix_coin")
    
    def _show_sync_status(self):
        """Show synchronization status"""
        print("Synchronization Status:")
        print(f"  Syncing: {self.sync_state['syncing']}")
        if self.sync_state['syncing']:
            print(f"  Progress: {self.sync_state['current_block']}/{self.sync_state['target_block']}")
            progress = (self.sync_state['current_block'] / self.sync_state['target_block']) * 100
            print(f"  Complete: {progress:.1f}%")
        print(f"  Connected Peers: {self.sync_state['peers_connected']}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutting down gracefully...")
    sys.exit(0)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAYONIX Blockchain Node")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--encryption-key', '-k', help='Encryption key for config file')
    parser.add_argument('--data-dir', '-d', help='Override data directory')
    parser.add_argument('--network', '-n', help='Override network type (mainnet/testnet/devnet)')
    parser.add_argument('--port', '-p', type=int, help='Override P2P port number')
    parser.add_argument('--no-network', action='store_true', help='Disable networking')
    parser.add_argument('--mining', action='store_true', help='Enable mining')
    parser.add_argument('--staking', action='store_true', help='Enable staking')
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and initialize node with config system
    node = RayonixNode(args.config, args.encryption_key)
    
    # Apply command-line overrides to config using config manager
    if args.data_dir:
        node.config_manager.set('database.db_path', args.data_dir)
    if args.network:
        node.config_manager.set('network.network_type', args.network.lower())
    if args.port:
        node.config_manager.set('network.listen_port', args.port)
    if args.no_network:
        node.config_manager.set('network.enabled', False)
    if args.mining:
        node.config_manager.set('mining.enabled', True)
    if args.staking:
        node.config_manager.set('consensus.consensus_type', 'pos')
    
    # Initialize node
    if not await node.initialize():
        print("Failed to initialize node")
        return 1
    
    # Start node
    if not await node.start():
        print("Failed to start node")
        return 1
    
    # Load command history
    if node.history_file.exists():
        readline.read_history_file(node.history_file)
    
    # Get current network from config
    current_network = node.get_config_value('network.network_type', 'mainnet')
    
    # Main command loop
    print(f"RAYONIX Blockchain Node started for {current_network.upper()} network")
    print("Type 'help' for commands, 'exit' to quit.")
    print(f"Using config: {node.config_manager.config_path or 'default settings'}")
    
    try:
        while True:
            try:
                command = input("rayonix> ").strip()
                if not command:
                    continue
                
                # Add to history
                node.command_history.append(command)
                readline.add_history(command)
                
                # Parse command
                parts = command.split()
                cmd = parts[0].lower()
                cmd_args = parts[1:]
                
                # Handle command
                should_continue = await node.handle_command(cmd, cmd_args)
                if not should_continue:
                    break
                    
            except EOFError:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Command error: {e}")
                print(f"Error: {e}")
                
    finally:
        # Save command history
        readline.write_history_file(node.history_file)
        await node.stop()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)