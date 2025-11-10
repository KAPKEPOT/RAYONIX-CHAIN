# rayonix_node/cli/base_commands/blockchain_commands.py

from typing import List, Dict, Any
from rayonix_node.cli.base_commands.base_command import BaseCommand


class BlockchainCommands(BaseCommand):
    """Blockchain query and exploration commands"""
    
    def execute_blockchain_info(self, args: List[str]) -> str:
        """Show detailed blockchain information"""
        try:
            blockchain_status = self.client.get_blockchain_status()
            node_info = self.client.get_detailed_info()
            
            response = "⛓️  BLOCKCHAIN INFORMATION\n"
            response += "────────────────────────────────\n"
            response += f"Block Height:      {blockchain_status.get('block_height', 0):,}\n"
            response += f"Network:           {node_info.get('network', 'Unknown').upper()}\n"
            response += f"Consensus:         {node_info.get('consensus', 'Unknown').upper()}\n"
            response += f"Difficulty:        {blockchain_status.get('difficulty', 0):.6f}\n"
            response += f"Total Transactions: {blockchain_status.get('total_transactions', 0):,}\n"
            response += f"Mempool Size:      {blockchain_status.get('mempool_size', 0):,}\n"
            response += f"Chain Work:        {blockchain_status.get('chain_work', 'Unknown')}\n"
            response += f"Best Block Hash:   {blockchain_status.get('best_block_hash', 'Unknown')[:20]}...\n"
            
            # Add sync information
            if blockchain_status.get('syncing', False):
                response += f"Sync Progress:     {blockchain_status.get('sync_progress', 0)}%\n"
                response += f"Sync Status:       ⚡ Syncing\n"
            else:
                response += f"Sync Status:       ✅ Fully synced\n"
            
            return response
        except Exception as e:
            return self._format_rpc_error(e)
    
    def execute_block(self, args: List[str]) -> str:
        """Show block information"""
        if not args:
            return "❌ Usage: block <height_or_hash>"
        
        try:
            block = self.client.get_block(args[0])
            if not block:
                return "❌ Block not found"
            
            return f"📦 BLOCK #{block.get('height', 'Unknown')}\n" \
                   f"────────────────────────────────\n" \
                   f"Hash:          {block.get('hash', 'Unknown')}\n" \
                   f"Previous Hash: {block.get('previous_hash', 'Genesis')}\n" \
                   f"Timestamp:     {self._format_timestamp(block.get('timestamp', 0))}\n" \
                   f"Transactions:  {len(block.get('transactions', []))}\n" \
                   f"Validator:     {block.get('validator', 'Unknown')}\n" \
                   f"Signature:     {block.get('signature', 'Unknown')[:20]}..."
        except Exception as e:
            return self._format_rpc_error(e)
    
    def execute_transaction(self, args: List[str]) -> str:
        """Show transaction information"""
        if not args:
            return "❌ Usage: transaction <hash>"
        
        try:
            transaction = self.client.get_transaction(args[0])
            if not transaction:
                return "❌ Transaction not found"
            
            return f"💸 TRANSACTION DETAILS\n" \
                   f"────────────────────────────────\n" \
                   f"Hash:      {transaction.get('hash', 'Unknown')}\n" \
                   f"Block:     {transaction.get('block_height', 'Unconfirmed')}\n" \
                   f"Timestamp: {self._format_timestamp(transaction.get('timestamp', 0))}\n" \
                   f"Inputs:    {len(transaction.get('inputs', []))}\n" \
                   f"Outputs:   {len(transaction.get('outputs', []))}\n" \
                   f"Amount:    {sum(out.get('amount', 0) for out in transaction.get('outputs', [])):,.6f} RYX"
        except Exception as e:
            return self._format_rpc_error(e)
    
    def execute_history(self, args: List[str]) -> str:
        """Transaction history"""
        try:
            count = int(args[0]) if args else 10
            transactions = self.client.get_transaction_history(count)
            
            if not transactions:
                return "📜 No transactions found"
            
            response = f"📜 RECENT TRANSACTIONS (last {len(transactions)})\n"
            response += "────────────────────────────────\n"
            for tx in transactions:
                tx_hash = tx.get('hash', 'Unknown')[:16] + '...'
                amount = tx.get('amount', 0)
                timestamp = tx.get('timestamp', 0)
                date = self._format_timestamp(timestamp)
                direction = "➡️  SENT" if amount < 0 else "⬅️  RECEIVED"
                response += f"{date} - {direction} - {abs(amount):,.6f} RYX\n"
                response += f"    Hash: {tx_hash}\n"
            
            return response
        except Exception as e:
            return self._format_rpc_error(e)