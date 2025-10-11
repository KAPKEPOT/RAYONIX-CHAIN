# api/jsonrpc_methods.py - JSON-RPC method implementations

import json
import time
import hashlib
from jsonrpcserver import method
from typing import Dict, List  # Add this import

from rayonix_node.utils.validators import validate_rayonix_address

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
        if not context.rayonix_chain._validate_transaction(tx_data):
            return {"error": "Invalid transaction"}
        
        # Add to mempool
        context.rayonix_chain._add_to_mempool(tx_data)
        
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
        
        # Calculate transaction hash
        tx_hash = hashlib.sha256(json.dumps(transaction, sort_keys=True).encode()).hexdigest()
        transaction["hash"] = tx_hash
        
        return {"result": transaction, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getblockcount(context) -> Dict:
    """JSON-RPC method to get current block count"""
    try:
        block_count = context.rayonix_chain.get_block_count()
        return {"result": block_count, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getblockhash(context, height: int) -> Dict:
    """JSON-RPC method to get block hash by height"""
    try:
        block_hash = context.rayonix_chain.get_block_hash(height)
        return {"result": block_hash, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getblock(context, hash_or_height) -> Dict:
    """JSON-RPC method to get block by hash or height"""
    try:
        if isinstance(hash_or_height, int):
            block = context.rayonix_chain.get_block_by_height(hash_or_height)
        else:
            block = context.rayonix_chain.get_block_by_hash(hash_or_height)
        
        if not block:
            return {"error": "Block not found"}
        
        return {"result": block, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def gettransaction(context, tx_hash: str) -> Dict:
    """JSON-RPC method to get transaction by hash"""
    try:
        transaction = context.rayonix_chain.get_transaction(tx_hash)
        if not transaction:
            return {"error": "Transaction not found"}
        
        return {"result": transaction, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getbalance(context, address: str = None) -> Dict:
    """JSON-RPC method to get balance for address or wallet"""
    try:
        if address:
            # Validate address format
            if not validate_rayonix_address(address):
                return {"error": "Invalid address format"}
            
            balance = context.rayonix_chain.get_address_balance(address)
        else:
            # Get wallet balance if no address specified
            if not context.wallet:
                return {"error": "No wallet available"}
            
            balance = context.wallet.get_balance()
        
        return {"result": balance, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getnewaddress(context, label: str = "") -> Dict:
    """JSON-RPC method to generate a new address"""
    try:
        if not context.wallet:
            return {"error": "No wallet available"}
        
        address = context.wallet.get_new_address()
        return {"result": address, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def listtransactions(context, count: int = 10, skip: int = 0) -> Dict:
    """JSON-RPC method to list recent transactions"""
    try:
        if not context.wallet:
            return {"error": "No wallet available"}
        
        transactions = context.wallet.get_transaction_history(count, skip)
        return {"result": transactions, "error": None}
    except Exception as e:
        return {"error": str(e)}

@method
async def getinfo(context) -> Dict:
    """JSON-RPC method to get node information"""
    try:
        info = {
            "version": "1.0.0",
            "protocolversion": 1,
            "blocks": context.rayonix_chain.get_block_count(),
            "connections": context.state_manager.get_sync_state().get('peers_connected', 0),
            "difficulty": context.rayonix_chain.get_difficulty(),
            "testnet": context.config_manager.get('network.network_type') == 'testnet',
            "balance": context.wallet.get_balance() if context.wallet else 0,
            "walletversion": 1,
            "timeoffset": 0,
            "keypoololdest": 0,
            "keypoolsize": 0,
            "paytxfee": 0,
            "relayfee": 0.0001,
            "errors": ""
        }
        return {"result": info, "error": None}
    except Exception as e:
        return {"error": str(e)}

def setup_jsonrpc_methods():
    """Setup all JSON-RPC methods"""
    # Methods are automatically registered via @method decorator
    pass