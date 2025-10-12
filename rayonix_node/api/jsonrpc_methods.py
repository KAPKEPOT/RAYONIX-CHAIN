# api/jsonrpc_methods.py - JSON-RPC method implementations

import json
import time
import hashlib
from jsonrpcserver import method, Result, Success, Error
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
        if hasattr(context.rayonix_chain, 'get_block_count'):
        	block_count = context.rayonix_chain.get_block_count()
        	return Success(block_count)
        else:
        	return Error(-32601, "Method not available")
        	
    except Exception as e:
    	logger.error(f"Error in getblockcount: {e}")
    	return Error(-32603, f"Internal error: {str(e)}")
 
@method
async def getblockhash(context, height: int) -> Result:
    """JSON-RPC method to get block hash by height"""
    try:
        if hasattr(context.rayonix_chain, 'get_block_hash'):
        	block_hash = context.rayonix_chain.get_block_hash(height)
        	if block_hash:
        		return Success(block_hash)
        	else:
        		return Error(-32602, f"Block at height {height} not found")
        else:
        	return Error(-32601, "Method not available")
    except Exception as e:
    	logger.error(f"Error in getblockhash: {e}")
    	return Error(-32603, f"Internal error: {str(e)}")

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
        # If no wallet exists, return 0 balance
        if not context.wallet:
        	return Success(0)
        
        if address:
        	# Validate address format
        	if not validate_rayonix_address(address):
        		return Error("Invalid address format")
        	
        	balance = context.rayonix_chain.get_address_balance(address)
        
        else:
        	# Get wallet balance
        	balance = context.wallet.get_balance()
        	
        return Success(balance)
    except Exception as e:
    	logger.error(f"Error getting balance: {e}")
    	return Error(str(e))

@method
async def getnewaddress(context, label: str = "") -> Dict:
    """JSON-RPC method to generate a new address"""
    try:
    	# Ensure wallet exists first
    	if not await context.create_wallet_if_not_exists():
    		return Error("Failed to initialize wallet")
    	
    	if not context.wallet:
    		return Error("No wallet available after initialization")
    	
    	address = context.wallet.get_new_address()
    	return Success(address)
    except Exception as e:
    	logger.error(f"Error generating address: {e}")
    	return Error(str(e))

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
        # Get block count safely
        block_count = 0
        if hasattr(context.rayonix_chain, 'get_block_count'):
        	block_count = context.rayonix_chain.get_block_count()
        
        # Get difficulty safely
        difficulty = 1
        if hasattr(context.rayonix_chain, 'get_difficulty'):
        	difficulty = context.rayonix_chain.get_difficulty()
        	
        # Get wallet balance safely
        balance = 0
        if hasattr(context, 'wallet') and context.wallet and hasattr(context.wallet, 'get_balance'):
        	balance = context.wallet.get_balance()
        	
        info = {
            "version": "1.0.0",
            "protocolversion": 1,
            "blocks": block_count,
            "connections": getattr(context.state_manager, 'peers_connected', 0),
            "difficulty": difficulty,
            "testnet": getattr(context.config_manager, 'network_type', 'testnet') == 'testnet',
            "balance": balance,
            "walletversion": 1,
            "timeoffset": 0,
            "keypoololdest": 0,
            "keypoolsize": 0,
            "paytxfee": 0,
            "relayfee": 0.0001,
            "errors": ""
        }
        return Success(info)
    except Exception as e:
        return Error(str(e))

def setup_jsonrpc_methods():
    """Setup all JSON-RPC methods"""
    # Methods are automatically registered via @method decorator
    pass