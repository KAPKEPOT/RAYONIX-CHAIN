# blockchain/transactions/coin_selection.py
import math
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from blockchain.models.utxo import UTXO

@dataclass
class CoinSelectionConfig:
    min_change_threshold: int = 1000  # Minimum change amount to create separate output
    dust_threshold: int = 546  # Below this amount is considered dust
    max_inputs: int = 100  # Maximum number of inputs to use
    privacy_weight: float = 0.7  # Weight for privacy considerations
    efficiency_weight: float = 0.3  # Weight for efficiency considerations

class CoinSelectionStrategy:
    """Base class for coin selection strategies"""
    
    def __init__(self, config: CoinSelectionConfig):
        self.config = config
    
    def select_coins(self, utxos: List[UTXO], target_amount: int, fee_rate: float = 1.0) -> Tuple[List[UTXO], int, int]:
        """Select coins for transaction (to be implemented by subclasses)"""
        raise NotImplementedError

class DefaultCoinSelection(CoinSelectionStrategy):
    """Default coin selection strategy (largest first)"""
    
    def select_coins(self, utxos: List[UTXO], target_amount: int, fee_rate: float = 1.0) -> Tuple[List[UTXO], int, int]:
        sorted_utxos = sorted(utxos, key=lambda x: x.amount, reverse=True)
        selected = []
        total = 0
        estimated_fee = self._estimate_fee(0, 2, fee_rate)  # Base fee for 2 outputs
        
        for utxo in sorted_utxos:
            if total >= target_amount + estimated_fee:
                break
            
            selected.append(utxo)
            total += utxo.amount
            
            # Update fee estimate with current input count
            estimated_fee = self._estimate_fee(len(selected), 2, fee_rate)
        
        change = total - target_amount - estimated_fee
        
        # Don't create change output if it's below threshold
        if change < self.config.min_change_threshold:
            # Add change to fee (overpay slightly)
            estimated_fee += change
            change = 0
        
        return selected, total, change
    
    def _estimate_fee(self, input_count: int, output_count: int, fee_rate: float) -> int:
        """Estimate transaction fee"""
        # Base transaction size + input size * input_count + output size * output_count
        base_size = 10  # bytes
        input_size = 150  # bytes per input
        output_size = 34  # bytes per output
        
        total_size = base_size + (input_size * input_count) + (output_size * output_count)
        return math.ceil(total_size * fee_rate)

class PrivacyCoinSelection(CoinSelectionStrategy):
    """Privacy-focused coin selection"""
    
    def select_coins(self, utxos: List[UTXO], target_amount: int, fee_rate: float = 1.0) -> Tuple[List[UTXO], int, int]:
        # Prefer older UTXOs for better privacy
        sorted_utxos = sorted(utxos, key=lambda x: x.age, reverse=True)
        selected = []
        total = 0
        estimated_fee = self._estimate_fee(0, 2, fee_rate)
        
        for utxo in sorted_utxos:
            if total >= target_amount + estimated_fee:
                break
            
            selected.append(utxo)
            total += utxo.amount
            estimated_fee = self._estimate_fee(len(selected), 2, fee_rate)
        
        change = total - target_amount - estimated_fee
        
        # For privacy, sometimes create change even if small
        if change > 0 and (change >= self.config.dust_threshold or random.random() < 0.3):
            # Create change output
            pass
        else:
            # No change output
            estimated_fee += change
            change = 0
        
        return selected, total, change
    
    def _estimate_fee(self, input_count: int, output_count: int, fee_rate: float) -> int:
        base_size = 10
        input_size = 150
        output_size = 34
        total_size = base_size + (input_size * input_count) + (output_size * output_count)
        return math.ceil(total_size * fee_rate)

class EfficiencyCoinSelection(CoinSelectionStrategy):
    """Efficiency-focused coin selection"""
    
    def select_coins(self, utxos: List[UTXO], target_amount: int, fee_rate: float = 1.0) -> Tuple[List[UTXO], int, int]:
        # Try to find single UTXO that covers amount
        for utxo in sorted(utxos, key=lambda x: x.amount, reverse=True):
            fee = self._estimate_fee(1, 1, fee_rate)  # 1 input, 1 output (no change)
            if utxo.amount >= target_amount + fee:
                change = utxo.amount - target_amount - fee
                if change < self.config.min_change_threshold:
                    # No change output
                    fee += change
                    change = 0
                return [utxo], utxo.amount, change
        
        # Fall back to default strategy
        default_strategy = DefaultCoinSelection(self.config)
        return default_strategy.select_coins(utxos, target_amount, fee_rate)
    
    def _estimate_fee(self, input_count: int, output_count: int, fee_rate: float) -> int:
        base_size = 10
        input_size = 150
        output_size = 34
        total_size = base_size + (input_size * input_count) + (output_size * output_count)
        return math.ceil(total_size * fee_rate)

class ConsolidationCoinSelection(CoinSelectionStrategy):
    """Consolidation strategy for combining many small UTXOs"""
    
    def select_coins(self, utxos: List[UTXO], target_amount: int, fee_rate: float = 1.0) -> Tuple[List[UTXO], int, int]:
        # Use all UTXOs (for consolidation)
        if target_amount == 0:  # Full consolidation
            selected = utxos[:self.config.max_inputs]
            total = sum(utxo.amount for utxo in selected)
            
            # Estimate fee for consolidation tx
            output_count = 1  # Single output for consolidation
            fee = self._estimate_fee(len(selected), output_count, fee_rate)
            
            if total > fee:
                change = total - fee
                return selected, total, change
            else:
                # Not enough to cover fee
                return [], 0, 0
        else:
            # Partial consolidation with target amount
            sorted_utxos = sorted(utxos, key=lambda x: x.amount)
            selected = []
            total = 0
            estimated_fee = self._estimate_fee(0, 2, fee_rate)
            
            for utxo in sorted_utxos:
                if len(selected) >= self.config.max_inputs:
                    break
                
                selected.append(utxo)
                total += utxo.amount
                estimated_fee = self._estimate_fee(len(selected), 2, fee_rate)
                
                if total >= target_amount + estimated_fee:
                    break
            
            change = total - target_amount - estimated_fee
            return selected, total, change
    
    def _estimate_fee(self, input_count: int, output_count: int, fee_rate: float) -> int:
        base_size = 10
        input_size = 150
        output_size = 34
        total_size = base_size + (input_size * input_count) + (output_size * output_count)
        return math.ceil(total_size * fee_rate)

class CoinSelectionManager:
    """Manager for coin selection strategies"""
    
    def __init__(self, config: CoinSelectionConfig):
        self.config = config
        self.strategies = {
            'default': DefaultCoinSelection(config),
            'privacy': PrivacyCoinSelection(config),
            'efficiency': EfficiencyCoinSelection(config),
            'consolidation': ConsolidationCoinSelection(config)
        }
    
    def select_coins(self, strategy_name: str, utxos: List[UTXO], 
                    target_amount: int, fee_rate: float = 1.0) -> Tuple[List[UTXO], int, int]:
        """Select coins using specified strategy"""
        strategy = self.strategies.get(strategy_name, self.strategies['default'])
        return strategy.select_coins(utxos, target_amount, fee_rate)
    
    def analyze_utxos(self, utxos: List[UTXO]) -> Dict[str, Any]:
        """Analyze UTXO set for optimal strategy selection"""
        if not utxos:
            return {'recommended_strategy': 'default', 'confidence': 0.0}
        
        total_amount = sum(utxo.amount for utxo in utxos)
        avg_amount = total_amount / len(utxos) if utxos else 0
        max_amount = max(utxo.amount for utxo in utxos) if utxos else 0
        min_amount = min(utxo.amount for utxo in utxos) if utxos else 0
        
        # Calculate strategy scores
        scores = {
            'default': 1.0,
            'privacy': 0.5,
            'efficiency': 0.0,
            'consolidation': 0.0
        }
        
        # Adjust scores based on UTXO characteristics
        if len(utxos) > 10 and avg_amount < 10000:  # Many small UTXOs
            scores['consolidation'] += 0.8
            scores['privacy'] += 0.2
        
        if max_amount > total_amount * 0.5:  # One large UTXO dominates
            scores['efficiency'] += 0.7
        
        if len(utxos) > 5:  # Good privacy opportunity
            scores['privacy'] += 0.3
        
        # Find best strategy
        best_strategy = max(scores.items(), key=lambda x: x[1])
        
        return {
            'recommended_strategy': best_strategy[0],
            'confidence': best_strategy[1] / max(sum(scores.values()), 1.0),
            'utxo_count': len(utxos),
            'total_amount': total_amount,
            'average_amount': avg_amount,
            'max_amount': max_amount,
            'min_amount': min_amount
        }