#!/usr/bin/env python3
"""
RAYONIX CLI Client - Production Ready
Main entry point for CLI that connects to rayonixd daemon via RPC
"""

import argparse
import json
import sys
import os
from typing import Dict, List, Any, Optional
import requests
import logging
from pathlib import Path

from rayonix_node.cli.history_manager import HistoryManager
from rayonix_node.cli.command_handler import CommandHandler
from rayonix_node.cli.interactive import run_interactive_mode
from rayonix_node.utils.helpers import configure_logging

logger = logging.getLogger("rayonix_cli")

class RayonixRPCClient:
    """RPC client for communicating with rayonixd daemon"""
    
    def __init__(self, api_url: str = "http://127.0.0.1:8545", timeout: int = 30):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Setup session for better connection handling
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Rayonix-CLI/1.0.0'
        })
    
    def call_jsonrpc(self, method: str, params: List[Any] = None, request_id: int = 1) -> Dict[str, Any]:
        """Make JSON-RPC call to daemon"""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": request_id
        }
        
        try:
            response = self.session.post(
                f"{self.api_url}/jsonrpc",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to daemon at {self.api_url}. Is rayonixd running?")
        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error: {e}")
        except Exception as e:
            raise Exception(f"RPC call failed: {e}")
    
    def call_rest_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> Any:
        """Make REST API call to daemon"""
        url = f"{self.api_url}/api/v1/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to daemon at {self.api_url}. Is rayonixd running?")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception(f"Endpoint not found: {endpoint}")
            else:
                error_data = e.response.json()
                raise Exception(f"API error: {error_data.get('detail', str(e))}")
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    # RPC method implementations
    def get_info(self) -> Dict[str, Any]:
        """Get node information via JSON-RPC"""
        result = self.call_jsonrpc("getinfo")
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    def get_balance(self, address: str = None) -> float:
        """Get balance for address or wallet"""
        result = self.call_jsonrpc("getbalance", [address] if address else [])
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    def get_new_address(self) -> str:
        """Generate new address"""
        result = self.call_jsonrpc("getnewaddress")
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    def get_block_count(self) -> int:
        """Get current block count"""
        result = self.call_jsonrpc("getblockcount")
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    def get_block(self, block_hash_or_height: str) -> Dict[str, Any]:
        """Get block by hash or height"""
        result = self.call_jsonrpc("getblock", [block_hash_or_height])
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction by hash"""
        result = self.call_jsonrpc("gettransaction", [tx_hash])
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    def list_transactions(self, count: int = 10, skip: int = 0) -> List[Dict]:
        """List recent transactions"""
        result = self.call_jsonrpc("listtransactions", [count, skip])
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    def create_raw_transaction(self, inputs: List[Dict], outputs: List[Dict]) -> Dict:
        """Create raw transaction"""
        result = self.call_jsonrpc("createrawtransaction", [inputs, outputs])
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    def send_raw_transaction(self, hex_tx: str) -> str:
        """Send raw transaction"""
        result = self.call_jsonrpc("sendrawtransaction", [hex_tx])
        if 'error' in result and result['error']:
            raise Exception(result['error'])
        return result['result']
    
    # REST API method implementations
    def send_transaction(self, to_address: str, amount: float, fee: float = 0.0) -> str:
        """Send transaction via REST API"""
        data = {
            "to": to_address,
            "amount": amount,
            "fee": fee
        }
        result = self.call_rest_api("/wallet/send", "POST", data)
        return result['tx_hash']
    
    def get_peers(self) -> List[Dict]:
        """Get connected peers via REST API"""
        result = self.call_rest_api("/node/peers")
        return result['peers']
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get node status via REST API"""
        return self.call_rest_api("/node/status")
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get blockchain status via REST API"""
        return self.call_rest_api("/blockchain/status")
    
    def get_wallet_addresses(self) -> List[str]:
        """Get wallet addresses via REST API"""
        result = self.call_rest_api("/wallet/addresses")
        return result['addresses']

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='RAYONIX CLI Client')
    parser.add_argument('--url', default='http://127.0.0.1:8545', 
                       help='Daemon API URL (default: http://127.0.0.1:8545)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--data-dir', default='~/.rayonix',
                       help='Data directory for CLI history (default: ~/.rayonix)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='WARNING', help='Log level')
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Info command
    subparsers.add_parser('info', help='Show node information')
    
    # Balance command
    balance_parser = subparsers.add_parser('balance', help='Show balance')
    balance_parser.add_argument('address', nargs='?', help='Address to check balance for')
    
    # Send command
    send_parser = subparsers.add_parser('send', help='Send transaction')
    send_parser.add_argument('address', help='Destination address')
    send_parser.add_argument('amount', type=float, help='Amount to send')
    send_parser.add_argument('--fee', type=float, default=0.0, help='Transaction fee')
    
    # Address command
    subparsers.add_parser('address', help='Generate new address')
    
    # Peers command
    subparsers.add_parser('peers', help='Show connected peers')
    
    # Block command
    block_parser = subparsers.add_parser('block', help='Get block information')
    block_parser.add_argument('hash_or_height', help='Block hash or height')
    
    # Transaction command
    tx_parser = subparsers.add_parser('transaction', help='Get transaction information')
    tx_parser.add_argument('hash', help='Transaction hash')
    
    # Status command
    subparsers.add_parser('status', help='Show node status')
    
    # Mempool command
    subparsers.add_parser('mempool', help='Show mempool information')
    
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
            from cli.interactive import run_interactive_mode
            run_interactive_mode(client, data_dir)
        else:
            # Single command mode
            from cli.command_handler import CommandHandler
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