# api/rest_routes.py - REST API route handlers for FastAPI

import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from rayonix_node.utils.validators import validate_rayonix_address

logger = logging.getLogger("rayonix_node.api")

# Create router
router = APIRouter(prefix="/api/v1", tags=["blockchain"])

# Pydantic models for request/response validation
class TransactionRequest(BaseModel):
    to: str
    amount: float
    fee: float = 0

class TransactionResponse(BaseModel):
    tx_hash: str

class BalanceResponse(BaseModel):
    balance: float

class AddressesResponse(BaseModel):
    addresses: List[str]

class BlockchainStatusResponse(BaseModel):
    height: int
    difficulty: float
    hashrate: float
    mempool_size: int
    network: str

# Dependency to get node instance
def get_node(request: Dict):
    return request["node"]

def setup_rest_routes(app, node):
    """Setup REST API routes for FastAPI"""
    
    # Store node in app state for access in endpoints
    app.state.node = node
    
    # Include the router
    app.include_router(router)
    
    logger.info("FastAPI REST routes setup completed successfully")

@router.get("/blockchain/status", response_model=BlockchainStatusResponse)
async def get_blockchain_status(request: Request):
    """Get blockchain status"""
    node = request.app.state.node
    try:
        status = {
            "height": node.rayonix_chain.get_block_count(),
            "difficulty": node.rayonix_chain.get_difficulty(),
            "hashrate": 0,
            "mempool_size": len(node.rayonix_chain.mempool),
            "network": node.config_manager.get('network.network_type')
        }
        return status
    except Exception as e:
        logger.error(f"Error getting blockchain status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/blockchain/block/{block_hash_or_height}")
async def get_block(block_hash_or_height: str, node: Any = Depends(lambda: router.node)):
    """Get block by hash or height"""
    try:
        if block_hash_or_height.isdigit():
            block = node.rayonix_chain.get_block_by_height(int(block_hash_or_height))
        else:
            block = node.rayonix_chain.get_block_by_hash(block_hash_or_height)
        
        if not block:
            raise HTTPException(status_code=404, detail="Block not found")
        
        return block
    except Exception as e:
        logger.error(f"Error getting block {block_hash_or_height}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/blockchain/transaction/{tx_hash}")
async def get_transaction(tx_hash: str, node: Any = Depends(lambda: router.node)):
    """Get transaction by hash"""
    try:
        transaction = node.rayonix_chain.get_transaction(tx_hash)
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return transaction
    except Exception as e:
        logger.error(f"Error getting transaction {tx_hash}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/balance", response_model=BalanceResponse)
async def get_wallet_balance(node: Any = Depends(lambda: router.node)):
    """Get wallet balance"""
    try:
        if not node.wallet:
            raise HTTPException(status_code=400, detail="Wallet not available")
        
        balance = node.wallet.get_balance()
        return {"balance": balance}
    except Exception as e:
        logger.error(f"Error getting wallet balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/addresses", response_model=AddressesResponse)
async def get_wallet_addresses(node: Any = Depends(lambda: router.node)):
    """Get wallet addresses"""
    try:
        if not node.wallet:
            raise HTTPException(status_code=400, detail="Wallet not available")
        
        addresses = node.wallet.get_addresses()
        return {"addresses": addresses}
    except Exception as e:
        logger.error(f"Error getting wallet addresses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wallet/send", response_model=TransactionResponse)
async def send_transaction(transaction: TransactionRequest, node: Any = Depends(lambda: router.node)):
    """Send transaction"""
    try:
        if not node.wallet:
            raise HTTPException(status_code=400, detail="Wallet not available")
        
        if not validate_rayonix_address(transaction.to):
            raise HTTPException(status_code=400, detail="Invalid address format")
        
        # Create and send transaction
        tx_hash = node.wallet.send(transaction.to, transaction.amount, transaction.fee)
        if not tx_hash:
            raise HTTPException(status_code=400, detail="Failed to send transaction")
        
        return {"tx_hash": tx_hash}
    except Exception as e:
        logger.error(f"Error sending transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/node/status")
async def get_node_status(node: Any = Depends(lambda: router.node)):
    """Get node status"""
    try:
        status = node.state_manager.get_state_summary()
        return status
    except Exception as e:
        logger.error(f"Error getting node status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/node/peers")
async def get_peers(node: Any = Depends(lambda: router.node)):
    """Get connected peers"""
    try:
        if not node.network:
            raise HTTPException(status_code=400, detail="Network not available")
        
        peers = await node.network.get_peers()
        return {"peers": peers}
    except Exception as e:
        logger.error(f"Error getting peers: {e}")
        raise HTTPException(status_code=500, detail=str(e))