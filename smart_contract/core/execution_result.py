# smart_contract/core/execution_result.py
import time
import json
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

@dataclass
class ExecutionResult:
    """Enhanced result of contract execution with detailed metrics"""
    success: bool
    return_value: Any = None
    gas_used: int = 0
    error: Optional[str] = None
    events: List[Dict] = field(default_factory=list)
    execution_time: float = 0
    memory_used: int = 0
    timestamp: float = field(default_factory=time.time)
    transaction_hash: Optional[str] = None
    call_stack: List[str] = field(default_factory=list)
    gas_breakdown: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'return_value': self.return_value,
            'gas_used': self.gas_used,
            'error': self.error,
            'events': self.events,
            'execution_time': self.execution_time,
            'memory_used': self.memory_used,
            'timestamp': self.timestamp,
            'transaction_hash': self.transaction_hash,
            'call_stack': self.call_stack,
            'gas_breakdown': self.gas_breakdown
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)
    
    def add_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Add an execution event"""
        self.events.append({
            'name': event_name,
            'data': event_data,
            'timestamp': time.time()
        })
    
    def add_gas_breakdown(self, operation: str, cost: int) -> None:
        """Add gas cost breakdown"""
        self.gas_breakdown[operation] = self.gas_breakdown.get(operation, 0) + cost
    
    def get_total_cost(self, gas_price: int) -> int:
        """Get total execution cost in base currency"""
        return self.gas_used * gas_price
    
    def get_cost_wei(self, gas_price: int) -> int:
        """Get execution cost in wei equivalent"""
        return self.get_total_cost(gas_price) * (10 ** 9)