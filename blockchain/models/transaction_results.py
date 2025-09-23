# blockchain/models/transaction_results.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time

@dataclass
class TransactionCreationResult:
    success: bool
    fee_estimate: int
    change_amount: int
    transaction: Optional['Transaction'] = None  # Transaction import will be handled later
    selected_utxos: List['UTXO'] = field(default_factory=list)  # UTXO import will be handled later
    error_message: Optional[str] = None
    total_input: int = 0
    total_output: int = 0
    network_fee: int = 0
    priority: str = "medium"
    size_bytes: int = 0
    creation_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction creation result to dictionary"""
        from blockchain.models.transaction import Transaction  # Avoid circular import
        from blockchain.models.utxo import UTXO  # Avoid circular import
        
        return {
            'success': self.success,
            'transaction': self.transaction.to_dict() if self.transaction else None,
            'fee_estimate': self.fee_estimate,
            'selected_utxos': [utxo.to_dict() for utxo in self.selected_utxos],
            'change_amount': self.change_amount,
            'error_message': self.error_message,
            'total_input': self.total_input,
            'total_output': self.total_output,
            'network_fee': self.network_fee,
            'priority': self.priority,
            'size_bytes': self.size_bytes,
            'creation_time': self.creation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionCreationResult':
        """Create transaction creation result from dictionary"""
        from blockchain.models.transaction import Transaction  # Avoid circular import
        from blockchain.models.utxo import UTXO  # Avoid circular import
        
        transaction = Transaction.from_dict(data['transaction']) if data['transaction'] else None
        utxos = [UTXO.from_dict(utxo) for utxo in data['selected_utxos']] if data['selected_utxos'] else []
        
        return cls(
            success=data['success'],
            fee_estimate=data['fee_estimate'],
            change_amount=data['change_amount'],
            transaction=transaction,
            selected_utxos=utxos,
            error_message=data.get('error_message'),
            total_input=data.get('total_input', 0),
            total_output=data.get('total_output', 0),
            network_fee=data.get('network_fee', 0),
            priority=data.get('priority', 'medium'),
            size_bytes=data.get('size_bytes', 0),
            creation_time=data.get('creation_time', time.time())
        )
    
    def calculate_efficiency(self) -> float:
        """Calculate transaction efficiency (output/input ratio)"""
        if self.total_input <= 0:
            return 0.0
        return self.total_output / self.total_input
    
    def get_fee_rate(self) -> float:
        """Calculate fee rate (fee per byte)"""
        if self.size_bytes <= 0:
            return 0.0
        return self.network_fee / self.size_bytes

@dataclass
class FeeEstimate:
    low: int
    medium: int
    high: int
    urgent: int
    timestamp: float
    confidence: float
    mempool_size: int
    block_target: int = 6  # Target blocks for confirmation
    fee_per_byte: Dict[str, float] = field(default_factory=dict)
    historical_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fee estimate to dictionary"""
        return {
            'low': self.low,
            'medium': self.medium,
            'high': self.high,
            'urgent': self.urgent,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'mempool_size': self.mempool_size,
            'block_target': self.block_target,
            'fee_per_byte': self.fee_per_byte,
            'historical_data': self.historical_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeeEstimate':
        """Create fee estimate from dictionary"""
        return cls(
            low=data['low'],
            medium=data['medium'],
            high=data['high'],
            urgent=data['urgent'],
            timestamp=data['timestamp'],
            confidence=data['confidence'],
            mempool_size=data['mempool_size'],
            block_target=data.get('block_target', 6),
            fee_per_byte=data.get('fee_per_byte', {}),
            historical_data=data.get('historical_data', [])
        )
    
    def get_fee_for_priority(self, priority: str) -> int:
        """Get fee amount for specific priority level"""
        priority = priority.lower()
        if priority == 'low':
            return self.low
        elif priority == 'medium':
            return self.medium
        elif priority == 'high':
            return self.high
        elif priority == 'urgent':
            return self.urgent
        else:
            return self.medium
    
    def is_stale(self, stale_threshold: int = 300) -> bool:
        """Check if fee estimate is stale"""
        return time.time() - self.timestamp > stale_threshold
    
    def calculate_confidence_interval(self) -> Dict[str, float]:
        """Calculate confidence interval for fee estimates"""
        import math
        
        std_dev = (self.high - self.low) / 4.0  # Approximation
        margin_of_error = 1.96 * std_dev / math.sqrt(self.mempool_size) if self.mempool_size > 0 else 0
        
        return {
            'low_ci': max(0, self.medium - margin_of_error),
            'high_ci': self.medium + margin_of_error,
            'margin_of_error': margin_of_error
        }