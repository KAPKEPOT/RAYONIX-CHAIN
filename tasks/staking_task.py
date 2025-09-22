# tasks/staking_task.py - Staking loop implementation

import asyncio
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger("rayonix_node.staking")

class StakingTask:
    """Handles Proof-of-Stake staking operations"""
    
    def __init__(self, node: 'RayonixNode'):
        self.node = node
        self.staking = False
        self.last_stake_time = 0
        self.stake_attempts = 0
    
    async def staking_loop(self):
        """Main staking loop for Proof-of-Stake consensus"""
        while self.node.running:
            try:
                # Check if staking is possible
                if not self._can_stake():
                    await asyncio.sleep(10)
                    continue
                
                # Start staking
                self.staking = True
                logger.info("Starting staking process")
                
                # Main staking loop
                while self.node.running and self._can_stake():
                    try:
                        # Attempt to create a stake
                        stake_result = await self._attempt_stake()
                        
                        if stake_result:
                            self.last_stake_time = time.time()
                            self.stake_attempts = 0
                            logger.info(f"Successfully created stake: {stake_result}")
                        else:
                            self.stake_attempts += 1
                            if self.stake_attempts > 10:
                                logger.warning("Multiple failed stake attempts")
                                await asyncio.sleep(30)  # Longer pause after failures
                            else:
                                await asyncio.sleep(1)  # Brief pause between attempts
                                
                    except Exception as e:
                        logger.error(f"Error in staking attempt: {e}")
                        await asyncio.sleep(5)
                
                # Staking ended
                self.staking = False
                logger.info("Staking process ended")
                
            except Exception as e:
                logger.error(f"Error in staking loop: {e}")
                await asyncio.sleep(10)
    
    def _can_stake(self) -> bool:
        """Check if staking is possible"""
        # Check if node is running
        if not self.node.running:
            return False
        
        # Check if wallet is available
        if not self.node.wallet:
            logger.debug("Cannot stake: No wallet available")
            return False
        
        # Check if we're using Proof-of-Stake
        if self.node.get_config_value('consensus.consensus_type') != 'pos':
            logger.debug("Cannot stake: Not using Proof-of-Stake consensus")
            return False
        
        # Check if node is synchronized
        if not self.node.state_manager.is_synced():
            logger.debug("Cannot stake: Node not synchronized")
            return False
        
        # Check wallet balance
        balance = self.node.wallet.get_balance()
        min_stake = self.node.get_config_value('consensus.min_stake', 1000)
        
        if balance < min_stake:
            logger.debug(f"Cannot stake: Balance {balance} below minimum stake {min_stake}")
            return False
        
        return True
    
    async def _attempt_stake(self) -> Optional[Dict]:
        """Attempt to create a stake"""
        try:
            # Get staking information from wallet
            staking_info = self.node.wallet.get_staking_info()
            if not staking_info or not staking_info.get('can_stake', False):
                return None
            
            # Create stake transaction
            stake_tx = self.node.wallet.create_stake_transaction()
            if not stake_tx:
                return None
            
            # Validate stake transaction
            if not self.node.rayonix_coin._validate_transaction(stake_tx):
                logger.warning("Invalid stake transaction created")
                return None
            
            # Add to mempool
            if not self.node.rayonix_coin._add_to_mempool(stake_tx):
                logger.warning("Failed to add stake transaction to mempool")
                return None
            
            # Broadcast to network
            if self.node.network:
                await self.node.network.broadcast_message('transaction', {'transaction': stake_tx})
            
            return {
                'tx_hash': stake_tx.get('hash'),
                'amount': staking_info.get('staking_amount', 0),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error creating stake: {e}")
            return None
    
    def get_staking_status(self) -> Dict:
        """Get current staking status"""
        return {
            'staking': self.staking,
            'last_stake_time': self.last_stake_time,
            'stake_attempts': self.stake_attempts,
            'can_stake': self._can_stake(),
            'wallet_balance': self.node.wallet.get_balance() if self.node.wallet else 0,
            'min_stake': self.node.get_config_value('consensus.min_stake', 1000)
        }
    
    async def stop_staking(self):
        """Stop staking process"""
        self.staking = False
        logger.info("Staking stopped")