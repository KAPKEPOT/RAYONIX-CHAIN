#!/usr/bin/env python3
"""
RAYONIX CLI Client - Production Ready
Main entry point for CLI that connects to rayonixd daemon via RPC
"""

import argparse
import json
import sys
import os
import time
from typing import Dict, List, Any, Optional
import requests
import logging
from pathlib import Path
import readline
import getpass
from datetime import datetime

# Import existing components
from rayonix_node.cli.history_manager import HistoryManager
from rayonix_node.cli.command_handler import CommandHandler
from rayonix_node.cli.interactive import run_interactive_mode
from rayonix_node.utils.helpers import configure_logging

logger = logging.getLogger("rayonix_cli")

class RayonixRPCClient:
    """RPC client"""
    
    def __init__(self, api_url: str = "http://127.0.0.1:52557", timeout: int = 30):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=100,
            max_retries=5,
            pool_block=True
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Rayonix-CLI/2.0.0',
            'Accept': 'application/json'
        })
        
        # Features
        self.cache = {}
        self.cache_timeout = 300
        self.performance_metrics = {
            'requests_made': 0,
            'cache_hits': 0,
            'average_response_time': 0
        }

    def call_jsonrpc(self, method: str, params: List[Any] = None, request_id: int = 1) -> Dict[str, Any]:
        """JSON-RPC with caching"""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": request_id
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.api_url}/jsonrpc",
                json=payload,
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics['requests_made'] += 1
            self.performance_metrics['average_response_time'] = (
                self.performance_metrics['average_response_time'] * (self.performance_metrics['requests_made'] - 1) + response_time
            ) / self.performance_metrics['requests_made']
            
            response.raise_for_status()
            result = response.json()
            
            if 'error' in result and result['error']:
                error_msg = result['error'].get('message', 'Unknown RPC error')
                raise Exception(f"RPC Error: {error_msg}")
                
            return result
            
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to daemon at {self.api_url}")
        except Exception as e:
            raise Exception(f"RPC communication failed: {e}")
    
    def call_rest_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> Any:
        """Make REST API call to daemon"""
        url = f"{self.api_url}/api/v1/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=self.timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=self.timeout)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    # Enhanced RPC methods
    def get_detailed_info(self) -> Dict[str, Any]:
        """Get detailed node information"""
        return self.call_rest_api("node/status")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        return self.call_rest_api("network/stats")
    
    def get_wallet_detailed_balance(self) -> Dict[str, Any]:
        """Get detailed wallet balance"""
        return self.call_rest_api("wallet/balance")
    
    def get_transaction_history(self, count: int = 50) -> List[Dict]:
        """Get transaction history"""
        return self.call_rest_api(f"wallet/transactions?count={count}")
    
    def get_staking_info(self) -> Dict[str, Any]:
        """Get staking information"""
        return self.call_rest_api("staking/info")
    
    def get_smart_contracts(self) -> List[Dict]:
        """Get deployed smart contracts"""
        return self.call_rest_api("contracts")
    
    def deploy_contract(self, code: str) -> str:
        """Deploy smart contract"""
        data = {"code": code}
        result = self.call_rest_api("contracts/deploy", "POST", data)
        return result['contract_address']
    
    def call_contract(self, address: str, function: str, args: List = None) -> Any:
        """Call contract function"""
        data = {
            "contract_address": address,
            "function": function,
            "args": args or []
        }
        return self.call_rest_api("contracts/call", "POST", data)

    # Wallet Management Methods
    def create_wallet(self, wallet_type: str = "hd", password: str = None) -> Dict[str, Any]:
        """Create a new wallet"""
        data = {
            "wallet_type": wallet_type,
            "password": password
        }
        return self.call_rest_api("wallet/create", "POST", data)
    
    def load_wallet(self, mnemonic: str, password: str = None) -> Dict[str, Any]:
        """Load wallet from mnemonic"""
        data = {
            "mnemonic": mnemonic,
            "password": password
        }
        return self.call_rest_api("wallet/load", "POST", data)
    
    def import_wallet(self, file_path: str, password: str = None) -> Dict[str, Any]:
        """Import wallet from backup file"""
        data = {
            "file_path": file_path,
            "password": password
        }
        return self.call_rest_api("wallet/import", "POST", data)
    
    def backup_wallet(self, file_path: str) -> Dict[str, Any]:
        """Backup wallet to file"""
        data = {"file_path": file_path}
        return self.call_rest_api("wallet/backup", "POST", data)
    
    def get_wallet_info(self) -> Dict[str, Any]:
        """Get wallet information"""
        return self.call_rest_api("wallet/info")
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration"""
        return self.call_rest_api("config")
    
    def stake_tokens(self, amount: float) -> Dict[str, Any]:
        """Stake tokens for validation"""
        data = {"amount": amount}
        return self.call_rest_api("staking/stake", "POST", data)
    
    def get_validators(self) -> List[Dict]:
        """Get validator list"""
        return self.call_rest_api("validators/list")
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get blockchain status"""
        return self.call_rest_api("blockchain/status")

    # Standard RPC methods
    def get_info(self) -> Dict[str, Any]:
        result = self.call_jsonrpc("getinfo")
        return result.get('result', {})
    
    def get_balance(self, address: str = None) -> float:
        params = [address] if address else []
        result = self.call_jsonrpc("getbalance", params)
        return result.get('result', 0)
    
    def get_new_address(self) -> str:
        result = self.call_jsonrpc("getnewaddress")
        return result.get('result', '')
    
    def get_block_count(self) -> int:
        result = self.call_jsonrpc("getblockcount")
        return result.get('result', 0)
    
    def get_block(self, block_hash_or_height: str) -> Dict[str, Any]:
        result = self.call_jsonrpc("getblock", [block_hash_or_height])
        return result.get('result', {})
    
    def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        result = self.call_jsonrpc("gettransaction", [tx_hash])
        return result.get('result', {})
    
    def list_transactions(self, count: int = 10, skip: int = 0) -> List[Dict]:
        result = self.call_jsonrpc("listtransactions", [count, skip])
        return result.get('result', [])
    
    def send_transaction(self, to_address: str, amount: float, fee: float = 0.0) -> str:
        data = {"to": to_address, "amount": amount, "fee": fee}
        result = self.call_rest_api("broadcast", "POST", data)
        return result.get('txid', '')
    
    def get_peers(self) -> List[Dict]:
        result = self.call_rest_api("network")
        return result.get('peers', [])
    
    def get_node_status(self) -> Dict[str, Any]:
        return self.call_rest_api("node/status")
    
    def get_wallet_addresses(self) -> List[str]:
        result = self.call_rest_api("wallet/addresses")
        return result.get('addresses', [])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get CLI performance metrics"""
        return self.performance_metrics

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='RAYONIX CLI Client')
    parser.add_argument('--url', default='http://127.0.0.1:52557', 
                       help='Daemon API URL (default: http://127.0.0.1:52557)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--data-dir', default='~/.rayonix',
                       help='Data directory for CLI history (default: ~/.rayonix)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='WARNING', help='Log level')
    
    # Enhanced command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Basic commands
    subparsers.add_parser('info', help='Show detailed node information')
    subparsers.add_parser('status', help='Show node status')
    
    # Wallet Management Commands
    create_parser = subparsers.add_parser('create-wallet', help='Create a new wallet')
    create_parser.add_argument('--type', default='hd', choices=['hd', 'legacy'], 
                              help='Wallet type (default: hd)')
    create_parser.add_argument('--password', help='Wallet password')
    
    load_parser = subparsers.add_parser('load-wallet', help='Load wallet from mnemonic')
    load_parser.add_argument('mnemonic', help='Mnemonic phrase')
    load_parser.add_argument('--password', help='Wallet password')
    
    import_parser = subparsers.add_parser('import-wallet', help='Import wallet from backup')
    import_parser.add_argument('file', help='Backup file path')
    import_parser.add_argument('--password', help='Wallet password')
    
    subparsers.add_parser('wallet-info', help='Show detailed wallet information')
    subparsers.add_parser('list-addresses', help='List all wallet addresses')
    
    backup_parser = subparsers.add_parser('backup-wallet', help='Backup wallet to file')
    backup_parser.add_argument('--file', help='Backup file path')
    
    # Wallet Operations
    balance_parser = subparsers.add_parser('balance', help='Show detailed balance')
    balance_parser.add_argument('address', nargs='?', help='Address to check balance for')
    
    send_parser = subparsers.add_parser('send', help='Send transaction')
    send_parser.add_argument('address', help='Destination address')
    send_parser.add_argument('amount', type=float, help='Amount to send')
    send_parser.add_argument('--fee', type=float, default=0.0, help='Transaction fee')
    
    subparsers.add_parser('address', help='Generate new address')
    
    # Network commands
    subparsers.add_parser('peers', help='Show connected peers')
    subparsers.add_parser('network', help='Show network statistics')
    
    # Blockchain commands
    subparsers.add_parser('blockchain-info', help='Show detailed blockchain information')
    subparsers.add_parser('sync-status', help='Show synchronization status')
    
    block_parser = subparsers.add_parser('block', help='Get block information')
    block_parser.add_argument('hash_or_height', help='Block hash or height')
    
    tx_parser = subparsers.add_parser('transaction', help='Get transaction information')
    tx_parser.add_argument('hash', help='Transaction hash')
    
    subparsers.add_parser('mempool', help='Show mempool information')
    
    # commands 
    subparsers.add_parser('staking', help='Show staking information')
    
    stake_parser = subparsers.add_parser('stake', help='Stake tokens for validation')
    stake_parser.add_argument('amount', type=float, help='Amount to stake')
    
    subparsers.add_parser('validator-info', help='Show validator information')
    subparsers.add_parser('contracts', help='List smart contracts')
    
    deploy_parser = subparsers.add_parser('deploy-contract', help='Deploy smart contract')
    deploy_parser.add_argument('code', help='Contract code')
    
    call_parser = subparsers.add_parser('call-contract', help='Call contract function')
    call_parser.add_argument('address', help='Contract address')
    call_parser.add_argument('function', help='Function name')
    call_parser.add_argument('args', nargs='*', help='Function arguments')
    
    # History and Stats
    history_parser = subparsers.add_parser('history', help='Show transaction history')
    history_parser.add_argument('--count', type=int, default=10, help='Number of transactions to show')
    
    subparsers.add_parser('stats', help='Show CLI statistics')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('key', nargs='?', help='Config key to show')
    
    # Interactive mode
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level=args.log_level, component='cli')
    
    # Expand data directory path
    data_dir = os.path.expanduser(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create RPC client
    client = RayonixRPCClient(args.url, args.timeout)
    
    try:
        if args.command == 'interactive' or not args.command:
            # Interactive mode
            run_interactive_mode(client, data_dir)
        else:
            # Single command handler
            handler = CommandHandler(client)
            
            command_args = []
            if args.command == 'balance' and hasattr(args, 'address') and args.address:
                command_args.append(args.address)
            elif args.command == 'send':
                command_args.extend([args.address, str(args.amount)])
                if args.fee > 0:
                    command_args.append(str(args.fee))
            elif args.command == 'block':
                command_args.append(args.hash_or_height)
            elif args.command == 'transaction':
                command_args.append(args.hash)
            elif args.command == 'create-wallet':
                if args.type:
                    command_args.append(args.type)
                if args.password:
                    command_args.append(args.password)
            elif args.command == 'load-wallet':
                command_args.append(args.mnemonic)
                if args.password:
                    command_args.append(args.password)
            elif args.command == 'import-wallet':
                command_args.append(args.file)
                if args.password:
                    command_args.append(args.password)
            elif args.command == 'backup-wallet' and hasattr(args, 'file') and args.file:
                command_args.append(args.file)
            elif args.command == 'deploy-contract':
                command_args.append(args.code)
            elif args.command == 'call-contract':
                command_args.extend([args.address, args.function] + args.args)
            elif args.command == 'stake':
                command_args.append(str(args.amount))
            elif args.command == 'history' and hasattr(args, 'count'):
                command_args.append(str(args.count))
            elif args.command == 'config' and hasattr(args, 'key') and args.key:
                command_args.append(args.key)
            
            result = handler.execute_command(args.command, command_args)
            print(result)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()