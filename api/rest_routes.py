# api/rest_routes.py - REST API route handlers

import json
from aiohttp import web
import logging

from utils.validators import validate_rayonix_address

logger = logging.getLogger("rayonix_node.api")

def setup_rest_routes(app, node):
    """Setup REST API routes"""
    
    if not app:
        raise ValueError("Application instance cannot be None")
    
    @app.get('/api/v1/blockchain/status')
    async def get_blockchain_status(request):
        """Get blockchain status"""
        try:
            status = {
                "height": node.rayonix_chain.get_block_count(),
                "difficulty": node.rayonix_chain.get_difficulty(),
                "hashrate": 0,  # Would calculate based on difficulty
                "mempool_size": len(node.rayonix_chain.mempool),
                "network": node.config_manager.get('network.network_type')
            }
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting blockchain status: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @app.get('/api/v1/blockchain/block/{block_hash_or_height}')
    async def get_block(request):
        """Get block by hash or height"""
        try:
            block_id = request.match_info['block_hash_or_height']
            if block_id.isdigit():
                block = node.rayonix_chain.get_block_by_height(int(block_id))
            else:
                block = node.rayonix_chain.get_block_by_hash(block_id)
            
            if not block:
                return web.json_response({"error": "Block not found"}, status=404)
            
            return web.json_response(block)
        except Exception as e:
            logger.error(f"Error getting block {request.match_info['block_hash_or_height']}: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @app.get('/api/v1/blockchain/transaction/{tx_hash}')
    async def get_transaction(request):
        """Get transaction by hash"""
        try:
            tx_hash = request.match_info['tx_hash']
            transaction = node.rayonix_chain.get_transaction(tx_hash)
            
            if not transaction:
                return web.json_response({"error": "Transaction not found"}, status=404)
            
            return web.json_response(transaction)
        except Exception as e:
            logger.error(f"Error getting transaction {request.match_info['tx_hash']}: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @app.get('/api/v1/wallet/balance')
    async def get_wallet_balance(request):
        """Get wallet balance"""
        try:
            if not node.wallet:
                return web.json_response({"error": "Wallet not available"}, status=400)
            
            balance = node.wallet.get_balance()
            return web.json_response({"balance": balance})
        except Exception as e:
            logger.error(f"Error getting wallet balance: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @app.get('/api/v1/wallet/addresses')
    async def get_wallet_addresses(request):
        """Get wallet addresses"""
        try:
            if not node.wallet:
                return web.json_response({"error": "Wallet not available"}, status=400)
            
            addresses = node.wallet.get_addresses()
            return web.json_response({"addresses": addresses})
        except Exception as e:
            logger.error(f"Error getting wallet addresses: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @app.post('/api/v1/wallet/send')
    async def send_transaction(request):
        """Send transaction"""
        try:
            if not node.wallet:
                return web.json_response({"error": "Wallet not available"}, status=400)
            
            data = await request.json()
            to_address = data.get('to')
            amount = data.get('amount')
            fee = data.get('fee', 0)
            
            if not to_address or not amount:
                return web.json_response({"error": "Missing parameters"}, status=400)
            
            if not validate_rayonix_address(to_address):
                return web.json_response({"error": "Invalid address"}, status=400)
            
            try:
                amount = float(amount)
                fee = float(fee)
            except ValueError:
                return web.json_response({"error": "Invalid amount or fee"}, status=400)
            
            # Create and send transaction
            tx_hash = node.wallet.send(to_address, amount, fee)
            if not tx_hash:
                return web.json_response({"error": "Failed to send transaction"}, status=400)
            
            return web.json_response({"tx_hash": tx_hash})
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @app.get('/api/v1/node/status')
    async def get_node_status(request):
        """Get node status"""
        try:
            status = node.state_manager.get_state_summary()
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting node status: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @app.get('/api/v1/node/peers')
    async def get_peers(request):
        """Get connected peers"""
        try:
            if not node.network:
                return web.json_response({"error": "Network not available"}, status=400)
            
            peers = await node.network.get_peers()
            return web.json_response({"peers": peers})
        except Exception as e:
            logger.error(f"Error getting peers: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    logger.info("REST routes setup completed successfully")