# network/message_handlers.py - Block/transaction message handlers

import logging
from typing import Dict, Any

logger = logging.getLogger("rayonix_node.network")

async def handle_block_message(message_data: Dict[str, Any], context: Any) -> bool:
    """Handle incoming block messages"""
    try:
        block = message_data.get('block')
        if not block:
            logger.warning("Received block message without block data")
            return False
        
        # Validate block structure
        if not _validate_block_structure(block):
            logger.warning("Received invalid block structure")
            return False
        
        # Check if we already have this block
        existing_block = context.rayonix_coin.get_block_by_hash(block.get('hash'))
        if existing_block:
            logger.debug(f"Already have block {block.get('hash')}")
            return True
        
        # Process the block
        success = context.rayonix_coin._process_block(block)
        if success:
            logger.info(f"Processed new block {block.get('hash')} from network")
            
            # Update sync state
            context.state_manager.update_sync_state(
                current_block=context.rayonix_coin.get_block_count()
            )
            
            return True
        else:
            logger.warning(f"Failed to process block {block.get('hash')}")
            return False
            
    except Exception as e:
        logger.error(f"Error handling block message: {e}")
        return False

async def handle_transaction_message(message_data: Dict[str, Any], context: Any) -> bool:
    """Handle incoming transaction messages"""
    try:
        transaction = message_data.get('transaction')
        if not transaction:
            logger.warning("Received transaction message without transaction data")
            return False
        
        # Validate transaction structure
        if not _validate_transaction_structure(transaction):
            logger.warning("Received invalid transaction structure")
            return False
        
        # Check if we already have this transaction
        existing_tx = context.rayonix_chain.get_transaction(transaction.get('hash'))
        if existing_tx:
            logger.debug(f"Already have transaction {transaction.get('hash')}")
            return True
        
        # Add to mempool
        success = context.rayonix_chain._add_to_mempool(transaction)
        if success:
            logger.info(f"Added transaction {transaction.get('hash')} to mempool")
            return True
        else:
            logger.warning(f"Failed to add transaction {transaction.get('hash')} to mempool")
            return False
            
    except Exception as e:
        logger.error(f"Error handling transaction message: {e}")
        return False

async def handle_peer_list_message(message_data: Dict[str, Any], context: Any) -> bool:
    """Handle incoming peer list messages"""
    try:
        peers = message_data.get('peers', [])
        if not peers:
            logger.debug("Received empty peer list")
            return True
        
        # Connect to new peers if we need more connections
        current_peers = await context.network.get_peers()
        if len(current_peers) < context.get_config_value('network.max_connections', 50):
            for peer in peers:
                if len(current_peers) >= context.get_config_value('network.max_connections', 50):
                    break
                
                peer_address = f"{peer.get('ip')}:{peer.get('port')}"
                try:
                    await context.network.connect_to_peer(peer_address)
                except Exception as e:
                    logger.debug(f"Failed to connect to peer {peer_address}: {e}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error handling peer list message: {e}")
        return False

async def handle_sync_request_message(message_data: Dict[str, Any], context: Any) -> bool:
    """Handle incoming sync request messages"""
    try:
        from_block = message_data.get('from_block', 0)
        to_block = message_data.get('to_block', 0)
        peer_id = message_data.get('peer_id')
        
        if not peer_id:
            logger.warning("Sync request without peer ID")
            return False
        
        # Validate block range
        current_height = context.rayonix_chain.get_block_count()
        if from_block < 0 or to_block > current_height or from_block > to_block:
            logger.warning(f"Invalid sync range requested: {from_block}-{to_block}")
            return False
        
        # Send requested blocks
        blocks_to_send = []
        for height in range(from_block, to_block + 1):
            block = context.rayonix_chain.get_block_by_height(height)
            if block:
                blocks_to_send.append(block)
        
        if blocks_to_send:
            await context.network.send_message(peer_id, 'blocks', {'blocks': blocks_to_send})
            logger.info(f"Sent {len(blocks_to_send)} blocks to {peer_id}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error handling sync request: {e}")
        return False

async def handle_ping_message(message_data: Dict[str, Any], context: Any) -> bool:
    """Handle incoming ping messages"""
    try:
        # Simply respond with pong
        peer_id = message_data.get('peer_id')
        if peer_id:
            await context.network.send_message(peer_id, 'pong', {'timestamp': message_data.get('timestamp')})
        
        return True
            
    except Exception as e:
        logger.error(f"Error handling ping message: {e}")
        return False

def _validate_block_structure(block: Dict[str, Any]) -> bool:
    """Validate block structure"""
    required_fields = ['hash', 'height', 'previous_hash', 'timestamp', 'transactions', 'nonce']
    return all(field in block for field in required_fields)

def _validate_transaction_structure(transaction: Dict[str, Any]) -> bool:
    """Validate transaction structure"""
    required_fields = ['hash', 'inputs', 'outputs', 'timestamp']
    return all(field in transaction for field in required_fields)