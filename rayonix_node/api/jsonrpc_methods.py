# api/jsonrpc_methods.py - Production Ready JSON-RPC Methods

import json
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal, getcontext

from jsonrpcserver import method, Result, Success, Error
import logging

from rayonix_node.utils.validators import validate_rayonix_address, validate_amount

logger = logging.getLogger("rayonix_node.api.jsonrpc")

# Set decimal precision for financial calculations
getcontext().prec = 18

# Constants for JSON-RPC
MAX_BLOCKS_PER_REQUEST = 100
MAX_TRANSACTIONS_PER_REQUEST = 1000
DEFAULT_CONFIRMATIONS = 6

@method
async def getblockcount(context) -> Result:
    """JSON-RPC method to get current block count"""
    try:
        if not hasattr(context.rayonix_chain, 'get_block_count'):
            return Error(-32601, "Method not available")
        
        block_count = context.rayonix_chain.get_block_count()
        return Success(block_count)
    except Exception as e:
        logger.error(f"Error in getblockcount: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getblockhash(context, height: int) -> Result:
    """JSON-RPC method to get block hash by height"""
    try:
        if not hasattr(context.rayonix_chain, 'get_block_hash'):
            return Error(-32601, "Method not available")
        
        if height < 0 or height > context.rayonix_chain.get_block_count():
            return Error(-32602, f"Block height {height} out of range")
        
        block_hash = context.rayonix_chain.get_block_hash(height)
        if not block_hash:
            return Error(-32602, f"Block at height {height} not found")
        
        return Success(block_hash)
    except Exception as e:
        logger.error(f"Error in getblockhash: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getblock(context, hash_or_height: Union[str, int], verbose: bool = True) -> Result:
    """JSON-RPC method to get block by hash or height"""
    try:
        if not hasattr(context.rayonix_chain, 'get_block_by_hash') or not hasattr(context.rayonix_chain, 'get_block_by_height'):
            return Error(-32601, "Method not available")
        
        # Get block
        if isinstance(hash_or_height, int):
            block = context.rayonix_chain.get_block_by_height(hash_or_height)
        else:
            block = context.rayonix_chain.get_block_by_hash(hash_or_height)
        
        if not block:
            return Error(-32602, "Block not found")
        
        if verbose:
            # Return full block data
            enhanced_block = block.copy() if isinstance(block, dict) else block.to_dict()
            
            # Add additional information
            enhanced_block['confirmations'] = context.rayonix_chain.get_block_count() - enhanced_block.get('height', 0) + 1
            enhanced_block['size'] = len(json.dumps(enhanced_block).encode('utf-8'))
            enhanced_block['transaction_count'] = len(enhanced_block.get('transactions', []))
            
            # Calculate total amount in block
            total_amount = 0
            for tx in enhanced_block.get('transactions', []):
                for output in tx.get('outputs', []):
                    total_amount += output.get('amount', 0)
            enhanced_block['total_amount'] = total_amount
            
            return Success(enhanced_block)
        else:
            # Return serialized block
            if isinstance(block, dict):
                block_data = json.dumps(block, sort_keys=True)
            else:
                block_data = block.serialize()
            return Success(block_data)
    except Exception as e:
        logger.error(f"Error in getblock: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getblockheader(context, hash_or_height: Union[str, int], verbose: bool = True) -> Result:
    """JSON-RPC method to get block header"""
    try:
        if not hasattr(context.rayonix_chain, 'get_block_header'):
            return Error(-32601, "Method not available")
        
        # Get block header
        if isinstance(hash_or_height, int):
            header = context.rayonix_chain.get_block_header_by_height(hash_or_height)
        else:
            header = context.rayonix_chain.get_block_header_by_hash(hash_or_height)
        
        if not header:
            return Error(-32602, "Block header not found")
        
        if verbose:
            return Success(header)
        else:
            # Return serialized header
            header_data = json.dumps(header, sort_keys=True)
            return Success(header_data)
    except Exception as e:
        logger.error(f"Error in getblockheader: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def gettransaction(context, tx_hash: str, include_watchonly: bool = False) -> Result:
    """JSON-RPC method to get transaction by hash"""
    try:
        if not hasattr(context.rayonix_chain, 'get_transaction'):
            return Error(-32601, "Method not available")
        
        transaction = context.rayonix_chain.get_transaction(tx_hash)
        if not transaction:
            # Check mempool
            if hasattr(context.rayonix_chain, 'mempool') and tx_hash in context.rayonix_chain.mempool:
                transaction = context.rayonix_chain.mempool[tx_hash]
                transaction['confirmations'] = 0
                transaction['blockhash'] = None
                transaction['blockindex'] = None
                transaction['blocktime'] = None
                transaction['timereceived'] = int(time.time())
            else:
                return Error(-32602, "Invalid or non-wallet transaction id")
        
        # Enhance transaction data
        enhanced_tx = transaction.copy()
        
        # Calculate confirmations
        if 'block_height' in transaction:
            confirmations = context.rayonix_chain.get_block_count() - transaction['block_height'] + 1
            enhanced_tx['confirmations'] = confirmations
            enhanced_tx['blockhash'] = context.rayonix_chain.get_block_hash(transaction['block_height'])
            enhanced_tx['blockindex'] = transaction.get('block_index', 0)
            enhanced_tx['blocktime'] = transaction.get('timestamp')
        else:
            enhanced_tx['confirmations'] = 0
            enhanced_tx['blockhash'] = None
            enhanced_tx['blockindex'] = None
            enhanced_tx['blocktime'] = None
        
        enhanced_tx['timereceived'] = transaction.get('timestamp', int(time.time()))
        enhanced_tx['time'] = enhanced_tx['timereceived']
        
        # Calculate amounts for wallet transactions
        if context.wallet and any(
            addr in context.wallet.get_addresses()
            for inp in transaction.get('inputs', [])
            for addr in [inp.get('address')] if addr
        ) or any(
            addr in context.wallet.get_addresses()
            for out in transaction.get('outputs', [])
            for addr in [out.get('address')] if addr
        ):
            
            # Calculate fee
            input_amount = sum(inp.get('amount', 0) for inp in transaction.get('inputs', []))
            output_amount = sum(out.get('amount', 0) for out in transaction.get('outputs', []))
            enhanced_tx['fee'] = input_amount - output_amount
            
            # Calculate amount (positive for received, negative for sent)
            wallet_addresses = context.wallet.get_addresses()
            amount = 0
            
            # Add outputs to wallet addresses
            for out in transaction.get('outputs', []):
                if out.get('address') in wallet_addresses:
                    amount += out.get('amount', 0)
            
            # Subtract inputs from wallet addresses
            for inp in transaction.get('inputs', []):
                if inp.get('address') in wallet_addresses:
                    amount -= inp.get('amount', 0)
            
            enhanced_tx['amount'] = amount
            enhanced_tx['details'] = []
            
            # Add input details
            for inp in transaction.get('inputs', []):
                if inp.get('address') in wallet_addresses:
                    enhanced_tx['details'].append({
                        'address': inp['address'],
                        'category': 'send',
                        'amount': -inp.get('amount', 0),
                        'vout': inp.get('output_index', 0)
                    })
            
            # Add output details
            for i, out in enumerate(transaction.get('outputs', [])):
                if out.get('address') in wallet_addresses:
                    enhanced_tx['details'].append({
                        'address': out['address'],
                        'category': 'receive',
                        'amount': out.get('amount', 0),
                        'vout': i
                    })
        
        return Success(enhanced_tx)
    except Exception as e:
        logger.error(f"Error in gettransaction: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getrawtransaction(context, tx_hash: str, verbose: bool = False) -> Result:
    """JSON-RPC method to get raw transaction"""
    try:
        if not hasattr(context.rayonix_chain, 'get_transaction'):
            return Error(-32601, "Method not available")
        
        transaction = context.rayonix_chain.get_transaction(tx_hash)
        if not transaction:
            return Error(-32602, "Transaction not found")
        
        if verbose:
            # Return enhanced transaction data
            enhanced_tx = transaction.copy()
            
            if 'block_height' in transaction:
                confirmations = context.rayonix_chain.get_block_count() - transaction['block_height'] + 1
                enhanced_tx['confirmations'] = confirmations
                enhanced_tx['blockhash'] = context.rayonix_chain.get_block_hash(transaction['block_height'])
                enhanced_tx['blocktime'] = transaction.get('timestamp')
            else:
                enhanced_tx['confirmations'] = 0
                enhanced_tx['blockhash'] = None
                enhanced_tx['blocktime'] = None
            
            enhanced_tx['time'] = transaction.get('timestamp', int(time.time()))
            enhanced_tx['hex'] = json.dumps(transaction)
            
            return Success(enhanced_tx)
        else:
            # Return serialized transaction
            tx_data = json.dumps(transaction, sort_keys=True)
            return Success(tx_data)
    except Exception as e:
        logger.error(f"Error in getrawtransaction: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def sendrawtransaction(context, hex_tx: str) -> Result:
    """JSON-RPC method to send a raw transaction"""
    try:
        if not hasattr(context.rayonix_chain, 'validate_transaction') or not hasattr(context.rayonix_chain, 'add_to_mempool'):
            return Error(-32601, "Method not available")
        
        # Decode hex transaction
        try:
            tx_data = json.loads(hex_tx)
        except json.JSONDecodeError:
            return Error(-32602, "Invalid transaction hex encoding")
        
        # Validate transaction structure
        required_fields = ['version', 'inputs', 'outputs', 'timestamp']
        if not all(field in tx_data for field in required_fields):
            return Error(-32602, "Invalid transaction structure")
        
        # Validate transaction
        if not context.rayonix_chain.validate_transaction(tx_data):
            return Error(-32602, "Transaction validation failed")
        
        # Calculate transaction hash
        tx_hash = hashlib.sha256(json.dumps(tx_data, sort_keys=True).encode()).hexdigest()
        tx_data['hash'] = tx_hash
        
        # Add to mempool
        context.rayonix_chain.add_to_mempool(tx_data)
        
        # Broadcast to network if available
        if hasattr(context, 'network') and context.network:
            await context.network.broadcast_transaction(tx_data)
        
        logger.info(f"Transaction broadcast: {tx_hash}")
        return Success(tx_hash)
    except Exception as e:
        logger.error(f"Error in sendrawtransaction: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def createrawtransaction(context, inputs: List[Dict], outputs: List[Dict], locktime: int = 0, replaceable: bool = False) -> Result:
    """JSON-RPC method to create a raw transaction"""
    try:
        # Validate inputs
        if not inputs or not outputs:
            return Error(-32602, "Inputs and outputs required")
        
        # Validate each input
        for i, inp in enumerate(inputs):
            if 'txid' not in inp or 'vout' not in inp:
                return Error(-32602, f"Input {i} missing txid or vout")
        
        # Validate each output
        for i, out in enumerate(outputs):
            if 'address' not in out and 'data' not in out:
                return Error(-32602, f"Output {i} must have address or data")
            if 'amount' in out and out['amount'] <= 0:
                return Error(-32602, f"Output {i} amount must be positive")
        
        # Create transaction
        transaction = {
            "version": 1,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": int(time.time()),
            "locktime": locktime
        }
        
        # Add replace-by-fee flag if enabled
        if replaceable:
            transaction['replaceable'] = True
        
        # Calculate transaction hash
        tx_hash = hashlib.sha256(json.dumps(transaction, sort_keys=True).encode()).hexdigest()
        transaction["hash"] = tx_hash
        
        # Serialize transaction
        hex_tx = json.dumps(transaction)
        
        return Success(hex_tx)
    except Exception as e:
        logger.error(f"Error in createrawtransaction: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def signrawtransaction(context, hex_tx: str, privkeys: List[str] = None, prevtxs: List[Dict] = None, sighashtype: str = "ALL") -> Result:
    """JSON-RPC method to sign a raw transaction"""
    try:
        if not context.wallet:
            return Error(-32603, "No wallet available")
        
        # Decode transaction
        try:
            tx_data = json.loads(hex_tx)
        except json.JSONDecodeError:
            return Error(-32602, "Invalid transaction hex")
        
        # Validate transaction structure
        if 'inputs' not in tx_data or 'outputs' not in tx_data:
            return Error(-32602, "Invalid transaction structure")
        
        # Sign transaction using wallet
        signed_tx = context.wallet.sign_transaction(tx_data)
        
        if not signed_tx:
            return Error(-32603, "Failed to sign transaction")
        
        # Verify signatures
        if not context.rayonix_chain.verify_transaction_signatures(signed_tx):
            return Error(-32602, "Transaction signature verification failed")
        
        # Return signed transaction
        signed_hex = json.dumps(signed_tx)
        
        return Success({
            "hex": signed_hex,
            "complete": True,
            "errors": []
        })
    except Exception as e:
        logger.error(f"Error in signrawtransaction: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getbalance(context, account: str = "*", minconf: int = 1, include_watchonly: bool = False) -> Result:
    """JSON-RPC method to get balance"""
    try:
        if not context.wallet:
            return Success(0.0)
        
        wallet = context.wallet
        
        # Get confirmed balance
        confirmed_balance = wallet.get_balance()
        
        # Get unconfirmed balance if minconf is 0
        total_balance = confirmed_balance
        if minconf == 0:
            unconfirmed_balance = wallet.get_unconfirmed_balance()
            total_balance += unconfirmed_balance
        
        return Success(float(total_balance))
    except Exception as e:
        logger.error(f"Error in getbalance: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getreceivedbyaddress(context, address: str, minconf: int = 1) -> Result:
    """JSON-RPC method to get total received by address"""
    try:
        if not context.wallet:
            return Success(0.0)
        
        if not validate_rayonix_address(address):
            return Error(-32602, "Invalid Rayonix address")
        
        # Check if address belongs to wallet
        wallet_addresses = context.wallet.get_addresses()
        if address not in wallet_addresses:
            return Success(0.0)
        
        # Calculate total received
        total_received = 0.0
        transactions = context.wallet.get_transaction_history(1000)  # Get recent transactions
        
        for tx in transactions:
            if tx.get('confirmations', 0) >= minconf:
                for output in tx.get('outputs', []):
                    if output.get('address') == address:
                        total_received += output.get('amount', 0)
        
        return Success(float(total_received))
    except Exception as e:
        logger.error(f"Error in getreceivedbyaddress: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getnewaddress(context, account: str = "", address_type: str = "legacy") -> Result:
    """JSON-RPC method to generate a new address"""
    try:
        if not context.wallet:
            return Error(-32603, "No wallet available")
        
        # Validate address type
        if address_type not in ["legacy", "p2sh-segwit", "bech32"]:
            return Error(-32602, "Invalid address type")
        
        # Generate new address
        new_address = context.wallet.get_new_address()
        
        if not new_address:
            return Error(-32603, "Failed to generate new address")
        
        return Success(new_address)
    except Exception as e:
        logger.error(f"Error in getnewaddress: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getaddressesbyaccount(context, account: str = "") -> Result:
    """JSON-RPC method to get addresses by account"""
    try:
        if not context.wallet:
            return Success([])
        
        addresses = context.wallet.get_addresses()
        return Success(addresses)
    except Exception as e:
        logger.error(f"Error in getaddressesbyaccount: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getaccountaddress(context, account: str = "") -> Result:
    """JSON-RPC method to get current address for account"""
    try:
        if not context.wallet:
            return Error(-32603, "No wallet available")
        
        addresses = context.wallet.get_addresses()
        if not addresses:
            return Error(-32603, "No addresses in wallet")
        
        return Success(addresses[0])
    except Exception as e:
        logger.error(f"Error in getaccountaddress: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def listtransactions(context, account: str = "*", count: int = 10, skip: int = 0, include_watchonly: bool = False) -> Result:
    """JSON-RPC method to list recent transactions"""
    try:
        if not context.wallet:
            return Success([])
        
        # Validate parameters
        if count < 1 or count > 1000:
            return Error(-32602, "Count must be between 1 and 1000")
        if skip < 0:
            return Error(-32602, "Skip must be non-negative")
        
        # Get transaction history
        transactions = context.wallet.get_transaction_history(count + skip)
        
        # Apply skip and limit
        transactions = transactions[skip:skip + count]
        
        # Enhance transaction data for JSON-RPC compatibility
        enhanced_transactions = []
        for tx in transactions:
            enhanced_tx = {
                "account": "",
                "address": "",
                "category": "",
                "amount": 0.0,
                "label": "",
                "vout": 0,
                "fee": 0.0,
                "confirmations": tx.get('confirmations', 0),
                "blockhash": tx.get('block_hash'),
                "blockindex": tx.get('block_index'),
                "blocktime": tx.get('block_timestamp'),
                "txid": tx.get('hash'),
                "time": tx.get('timestamp', 0),
                "timereceived": tx.get('timestamp', 0),
                "bip125-replaceable": "no"
            }
            
            # Determine category and amount
            wallet_addresses = context.wallet.get_addresses()
            amount = 0.0
            
            # Check if transaction involves wallet addresses
            for inp in tx.get('inputs', []):
                if inp.get('address') in wallet_addresses:
                    amount -= inp.get('amount', 0)
                    enhanced_tx['address'] = inp['address']
                    enhanced_tx['category'] = 'send'
            
            for out in tx.get('outputs', []):
                if out.get('address') in wallet_addresses:
                    amount += out.get('amount', 0)
                    if enhanced_tx['category'] != 'send':  # Not already categorized as send
                        enhanced_tx['address'] = out['address']
                        enhanced_tx['category'] = 'receive'
            
            enhanced_tx['amount'] = float(amount)
            
            # Calculate fee for sent transactions
            if enhanced_tx['category'] == 'send':
                input_amount = sum(inp.get('amount', 0) for inp in tx.get('inputs', []))
                output_amount = sum(out.get('amount', 0) for out in tx.get('outputs', []))
                enhanced_tx['fee'] = float(input_amount - output_amount)
            
            enhanced_transactions.append(enhanced_tx)
        
        return Success(enhanced_transactions)
    except Exception as e:
        logger.error(f"Error in listtransactions: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def listaccounts(context, minconf: int = 1, include_watchonly: bool = False) -> Result:
    """JSON-RPC method to list accounts"""
    try:
        if not context.wallet:
            return Success({})
        
        # For simplicity, return a single account
        balance = context.wallet.get_balance()
        
        return Success({
            "": float(balance)
        })
    except Exception as e:
        logger.error(f"Error in listaccounts: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getwalletinfo(context) -> Result:
    """JSON-RPC method to get wallet information"""
    try:
        if not context.wallet:
            return Success({
                "walletname": "",
                "walletversion": 1,
                "balance": 0.0,
                "unconfirmed_balance": 0.0,
                "immature_balance": 0.0,
                "txcount": 0,
                "keypoololdest": 0,
                "keypoolsize": 0,
                "keypoolsize_hd_internal": 0,
                "unlocked_until": 0,
                "paytxfee": 0.0,
                "hdseedid": None,
                "private_keys_enabled": True,
                "avoid_reuse": False,
                "scanning": False
            })
        
        wallet = context.wallet
        addresses = wallet.get_addresses()
        
        info = {
            "walletname": wallet.get_wallet_id(),
            "walletversion": 1,
            "balance": float(wallet.get_balance()),
            "unconfirmed_balance": float(wallet.get_unconfirmed_balance()),
            "immature_balance": 0.0,
            "txcount": len(wallet.get_transaction_history(10000)),
            "keypoololdest": int(time.time()),
            "keypoolsize": len(addresses),
            "keypoolsize_hd_internal": 0,
            "unlocked_until": 0,
            "paytxfee": 0.0,
            "hdseedid": wallet.get_wallet_id() if hasattr(wallet, 'get_wallet_id') else None,
            "private_keys_enabled": not wallet.is_encrypted(),
            "avoid_reuse": False,
            "scanning": False
        }
        
        # Add HD wallet specific info
        if hasattr(wallet, 'get_master_key_id'):
            info['hdseedid'] = wallet.get_master_key_id()
            info['keypoolsize_hd_internal'] = len(addresses)
        
        return Success(info)
    except Exception as e:
        logger.error(f"Error in getwalletinfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getnetworkinfo(context) -> Result:
    """JSON-RPC method to get network information"""
    try:
        network_info = {
            "version": 1000000,
            "subversion": "/Rayonix:1.0.0/",
            "protocolversion": 70015,
            "localservices": "000000000000040d",
            "localrelay": True,
            "timeoffset": 0,
            "networkactive": True,
            "connections": getattr(context.state_manager, 'peers_connected', 0),
            "networks": [
                {
                    "name": "ipv4",
                    "limited": False,
                    "reachable": True,
                    "proxy": "",
                    "proxy_randomize_credentials": False
                }
            ],
            "relayfee": 0.00001000,
            "incrementalfee": 0.00001000,
            "localaddresses": [],
            "warnings": ""
        }
        
        return Success(network_info)
    except Exception as e:
        logger.error(f"Error in getnetworkinfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getmempoolinfo(context) -> Result:
    """JSON-RPC method to get mempool information"""
    try:
        if not hasattr(context.rayonix_chain, 'mempool'):
            return Success({
                "size": 0,
                "bytes": 0,
                "usage": 0,
                "maxmempool": 300000000,
                "mempoolminfee": 0.00001000,
                "minrelaytxfee": 0.00001000
            })
        
        mempool = context.rayonix_chain.mempool
        
        # Calculate mempool statistics
        total_bytes = sum(len(json.dumps(tx).encode('utf-8')) for tx in mempool.values())
        total_usage = total_bytes * 2  # Estimated memory usage
        
        fees = [tx.get('fee', 0) for tx in mempool.values() if 'fee' in tx]
        mempool_min_fee = min(fees) if fees else 0.00001000
        
        info = {
            "size": len(mempool),
            "bytes": total_bytes,
            "usage": total_usage,
            "maxmempool": 300000000,
            "mempoolminfee": float(mempool_min_fee),
            "minrelaytxfee": 0.00001000
        }
        
        return Success(info)
    except Exception as e:
        logger.error(f"Error in getmempoolinfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getmininginfo(context) -> Result:
    """JSON-RPC method to get mining information"""
    try:
        chain = context.rayonix_chain
        
        mining_info = {
            "blocks": chain.get_block_count(),
            "currentblocksize": 0,
            "currentblocktx": 0,
            "difficulty": float(chain.get_difficulty()),
            "networkhashps": chain.get_network_hashrate(),
            "pooledtx": len(chain.mempool),
            "chain": context.config_manager.get('network.network_type', 'testnet'),
            "warnings": ""
        }
        
        return Success(mining_info)
    except Exception as e:
        logger.error(f"Error in getmininginfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getstakinginfo(context) -> Result:
    """JSON-RPC method to get staking information"""
    try:
        # Default staking info
        staking_info = {
            "enabled": False,
            "staking": False,
            "errors": "",
            "currentblocksize": 0,
            "currentblocktx": 0,
            "pooledtx": 0,
            "difficulty": 0.0,
            "search-interval": 0,
            "weight": 0,
            "netstakeweight": 0,
            "expectedtime": 0
        }
        
        # Get real staking info if available
        if hasattr(context, 'staking_manager') and context.staking_manager:
            try:
                real_info = context.staking_manager.get_staking_info()
                staking_info.update(real_info)
            except Exception as e:
                logger.warning(f"Could not get staking info: {e}")
        
        return Success(staking_info)
    except Exception as e:
        logger.error(f"Error in getstakinginfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getpeerinfo(context) -> Result:
    """JSON-RPC method to get peer information"""
    try:
        if not hasattr(context, 'network') or not context.network:
            return Success([])
        
        peers = await context.network.get_peers()
        
        enhanced_peers = []
        for peer in peers:
            enhanced_peer = {
                "id": peer.get('id', 0),
                "addr": peer.get('address', ''),
                "addrbind": peer.get('address', ''),
                "services": "000000000000040d",
                "relaytxes": True,
                "lastsend": peer.get('last_send', 0),
                "lastrecv": peer.get('last_receive', 0),
                "bytessent": peer.get('bytes_sent', 0),
                "bytesrecv": peer.get('bytes_received', 0),
                "conntime": peer.get('connected_since', 0),
                "timeoffset": 0,
                "pingtime": peer.get('ping_time', 0),
                "minping": peer.get('min_ping', 0),
                "version": peer.get('version', 70015),
                "subver": peer.get('user_agent', '/Rayonix:1.0.0/'),
                "inbound": peer.get('inbound', False),
                "addnode": False,
                "startingheight": peer.get('starting_height', 0),
                "banscore": 0,
                "synced_headers": -1,
                "synced_blocks": -1,
                "inflight": [],
                "whitelisted": False,
                "permissions": [],
                "minfeefilter": 0.00001000,
                "bytessent_per_msg": {},
                "bytesrecv_per_msg": {}
            }
            enhanced_peers.append(enhanced_peer)
        
        return Success(enhanced_peers)
    except Exception as e:
        logger.error(f"Error in getpeerinfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getinfo(context) -> Result:
    """JSON-RPC method to get node information"""
    try:
        chain = context.rayonix_chain
        
        info = {
            "version": 1000000,
            "protocolversion": 70015,
            "walletversion": 1,
            "balance": 0.0,
            "blocks": chain.get_block_count(),
            "timeoffset": 0,
            "connections": getattr(context.state_manager, 'peers_connected', 0),
            "proxy": "",
            "difficulty": float(chain.get_difficulty()),
            "testnet": context.config_manager.get('network.network_type', 'testnet') == 'testnet',
            "keypoololdest": int(time.time()),
            "keypoolsize": 0,
            "paytxfee": 0.0,
            "relayfee": 0.00001000,
            "errors": "",
            "maxblocksize": 1000000,
            "maxmempool": 300000000
        }
        
        # Add wallet balance if available
        if context.wallet:
            info["balance"] = float(context.wallet.get_balance())
            info["keypoolsize"] = len(context.wallet.get_addresses())
            info["walletversion"] = 1
        
        return Success(info)
    except Exception as e:
        logger.error(f"Error in getinfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def validateaddress(context, address: str) -> Result:
    """JSON-RPC method to validate an address"""
    try:
        is_valid = validate_rayonix_address(address)
        
        response = {
            "isvalid": is_valid,
            "address": address if is_valid else "",
            "scriptPubKey": "",
            "isscript": False,
            "iswitness": False
        }
        
        # If valid and wallet exists, check if address is in wallet
        if is_valid and context.wallet:
            wallet_addresses = context.wallet.get_addresses()
            response["ismine"] = address in wallet_addresses
            response["iswatchonly"] = False
        else:
            response["ismine"] = False
            response["iswatchonly"] = False
        
        return Success(response)
    except Exception as e:
        logger.error(f"Error in validateaddress: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getblockchaininfo(context) -> Result:
    """JSON-RPC method to get blockchain information"""
    try:
        chain = context.rayonix_chain
        
        blockchain_info = {
            "chain": context.config_manager.get('network.network_type', 'testnet'),
            "blocks": chain.get_block_count(),
            "headers": chain.get_block_count(),
            "bestblockhash": chain.get_best_block_hash(),
            "difficulty": float(chain.get_difficulty()),
            "mediantime": chain.get_median_time(),
            "verificationprogress": chain.get_sync_progress(),
            "initialblockdownload": chain.is_syncing(),
            "chainwork": chain.get_chain_work(),
            "size_on_disk": chain.get_chain_size(),
            "pruned": False,
            "pruneheight": 0,
            "automatic_pruning": False,
            "prune_target_size": 0,
            "softforks": {},
            "warnings": ""
        }
        
        return Success(blockchain_info)
    except Exception as e:
        logger.error(f"Error in getblockchaininfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def estimatesmartfee(context, conf_target: int, estimate_mode: str = "CONSERVATIVE") -> Result:
    """JSON-RPC method to estimate smart fee"""
    try:
        if conf_target < 1 or conf_target > 1008:
            return Error(-32602, "Invalid conf_target, must be between 1 and 1008")
        
        if estimate_mode not in ["UNSET", "ECONOMICAL", "CONSERVATIVE"]:
            return Error(-32602, "Invalid estimate_mode")
        
        # Simple fee estimation based on mempool size
        mempool_size = len(context.rayonix_chain.mempool)
        base_fee = 0.00001000  # Base fee rate
        
        # Increase fee based on mempool size and confirmation target
        if conf_target <= 2:
            # High priority - next block
            fee_rate = base_fee * (1 + (mempool_size / 1000))
        elif conf_target <= 6:
            # Medium priority - within 6 blocks
            fee_rate = base_fee * (1 + (mempool_size / 2000))
        else:
            # Low priority - more than 6 blocks
            fee_rate = base_fee
        
        # Apply estimate mode multiplier
        if estimate_mode == "CONSERVATIVE":
            fee_rate *= 1.5
        elif estimate_mode == "ECONOMICAL":
            fee_rate *= 0.8
        
        return Success({
            "feerate": float(fee_rate),
            "blocks": conf_target
        })
    except Exception as e:
        logger.error(f"Error in estimatesmartfee: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getblocktemplate(context, template_request: Dict = None) -> Result:
    """JSON-RPC method to get block template for mining"""
    try:
        if template_request is None:
            template_request = {}
        
        chain = context.rayonix_chain
        
        # Get current best block
        best_block = chain.get_best_block()
        if not best_block:
            return Error(-32603, "Cannot get best block")
        
        # Create block template
        template = {
            "version": 1,
            "previousblockhash": best_block.get('hash'),
            "transactions": [],
            "coinbaseaux": {
                "flags": "rayonix"
            },
            "coinbasevalue": 50.0,  # Block reward
            "target": "00000000ffff0000000000000000000000000000000000000000000000000000",
            "mintime": int(time.time()),
            "mutable": [
                "time",
                "transactions",
                "prevblock"
            ],
            "noncerange": "00000000ffffffff",
            "sigoplimit": 20000,
            "sizelimit": 1000000,
            "curtime": int(time.time()),
            "bits": "1d00ffff",
            "height": chain.get_block_count() + 1
        }
        
        # Add transactions from mempool
        mempool_txs = list(context.rayonix_chain.mempool.values())[:100]  # Limit to 100 transactions
        for tx in mempool_txs:
            template["transactions"].append({
                "data": json.dumps(tx),
                "txid": tx.get('hash'),
                "hash": tx.get('hash'),
                "fee": tx.get('fee', 0),
                "sigops": 1
            })
        
        return Success(template)
    except Exception as e:
        logger.error(f"Error in getblocktemplate: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

# Smart Contract JSON-RPC Methods
@method
async def deploycontract(context, code: str, name: str = "", initial_balance: float = 0.0) -> Result:
    """JSON-RPC method to deploy a smart contract"""
    try:
        if not hasattr(context, 'contract_manager') or not context.contract_manager:
            return Error(-32601, "Smart contracts not enabled")
        
        if not context.wallet:
            return Error(-32603, "No wallet available")
        
        # Deploy contract
        result = await context.contract_manager.deploy_contract(
            context.wallet,
            code,
            name,
            initial_balance
        )
        
        if not result.get('success'):
            return Error(-32603, result.get('error', 'Contract deployment failed'))
        
        return Success({
            "contract_address": result['contract_address'],
            "transaction_hash": result['transaction_hash'],
            "gas_used": result.get('gas_used', 0)
        })
    except Exception as e:
        logger.error(f"Error in deploycontract: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def callcontract(context, contract_address: str, function: str, args: List = None, value: float = 0.0) -> Result:
    """JSON-RPC method to call a contract function"""
    try:
        if not hasattr(context, 'contract_manager') or not context.contract_manager:
            return Error(-32601, "Smart contracts not enabled")
        
        if not context.wallet:
            return Error(-32603, "No wallet available")
        
        if args is None:
            args = []
        
        # Call contract
        result = await context.contract_manager.call_contract(
            context.wallet,
            contract_address,
            function,
            args,
            value
        )
        
        if not result.get('success'):
            return Error(-32603, result.get('error', 'Contract call failed'))
        
        return Success({
            "result": result.get('return_value'),
            "transaction_hash": result.get('transaction_hash'),
            "gas_used": result.get('gas_used', 0)
        })
    except Exception as e:
        logger.error(f"Error in callcontract: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

@method
async def getcontractinfo(context, contract_address: str) -> Result:
    """JSON-RPC method to get contract information"""
    try:
        if not hasattr(context, 'contract_manager') or not context.contract_manager:
            return Error(-32601, "Smart contracts not enabled")
        
        contract_info = await context.contract_manager.get_contract_info(contract_address)
        if not contract_info:
            return Error(-32602, "Contract not found")
        
        return Success(contract_info)
    except Exception as e:
        logger.error(f"Error in getcontractinfo: {e}")
        return Error(-32603, f"Internal error: {str(e)}")

def setup_jsonrpc_methods():
    """Setup all JSON-RPC methods"""
    # Methods are automatically registered via @method decorator
    logger.info("JSON-RPC methods setup completed")