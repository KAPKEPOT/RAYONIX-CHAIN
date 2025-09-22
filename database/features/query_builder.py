from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging

from database.utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)

class QueryOperator(Enum):
    EQ = auto()
    NE = auto()
    GT = auto()
    GTE = auto()
    LT = auto()
    LTE = auto()
    IN = auto()
    NIN = auto()
    EXISTS = auto()
    CONTAINS = auto()
    STARTS_WITH = auto()
    ENDS_WITH = auto()

@dataclass
class QueryCondition:
    field: str
    operator: QueryOperator
    value: Any

class QueryBuilder:
    """Advanced query builder with support for complex queries"""
    
    def __init__(self, db):
        self.db = db
        self.conditions: List[QueryCondition] = []
        self.limit: Optional[int] = None
        self.offset: Optional[int] = None
        self.sort_field: Optional[str] = None
        self.sort_reverse: bool = False
        self.projection: Optional[List[str]] = None
    
    def where(self, field: str, operator: QueryOperator, value: Any) -> 'QueryBuilder':
        """Add a condition to the query"""
        self.conditions.append(QueryCondition(field, operator, value))
        return self
    
    def limit(self, limit: int) -> 'QueryBuilder':
        """Set result limit"""
        self.limit = limit
        return self
    
    def offset(self, offset: int) -> 'QueryBuilder':
        """Set result offset"""
        self.offset = offset
        return self
    
    def sort(self, field: str, reverse: bool = False) -> 'QueryBuilder':
        """Set sort field and direction"""
        self.sort_field = field
        self.sort_reverse = reverse
        return self
    
    def select(self, fields: List[str]) -> 'QueryBuilder':
        """Set projection fields"""
        self.projection = fields
        return self
    
    def execute(self) -> List[Any]:
        """Execute the query"""
        try:
            # For simple queries, use existing indexes
            if len(self.conditions) == 1 and self._can_use_index(self.conditions[0]):
                return self._execute_index_query(self.conditions[0])
            
            # For complex queries, use filtering
            return self._execute_filter_query()
            
        except Exception as e:
            raise DatabaseError(f"Query execution failed: {e}")
    
    def _can_use_index(self, condition: QueryCondition) -> bool:
        """Check if condition can use an index"""
        index_name = f"index_{condition.field}"
        return index_name in self.db.indexes
    
    def _execute_index_query(self, condition: QueryCondition) -> List[Any]:
        """Execute query using index"""
        index_name = f"index_{condition.field}"
        index = self.db.indexes[index_name]
        
        if condition.operator == QueryOperator.EQ:
            return index.query(condition.value, self.limit or 1000, self.offset or 0)
        
        # Other operators would require range queries
        return []
    
    def _execute_filter_query(self) -> List[Any]:
        """Execute query by filtering all documents"""
        results = []
        count = 0
        skipped = 0
        
        for key, value in self.db.iterate():
            if self.offset and skipped < self.offset:
                skipped += 1
                continue
            
            if self._matches_all_conditions(value):
                if self.projection:
                    results.append(self._apply_projection(value))
                else:
                    results.append(value)
                
                count += 1
                if self.limit and count >= self.limit:
                    break
        
        if self.sort_field:
            results.sort(
                key=lambda x: x.get(self.sort_field) if isinstance(x, dict) else getattr(x, self.sort_field, None),
                reverse=self.sort_reverse
            )
        
        return results
    
    def _matches_all_conditions(self, value: Any) -> bool:
        """Check if value matches all conditions"""
        for condition in self.conditions:
            if not self._matches_condition(value, condition):
                return False
        return True
    
    def _matches_condition(self, value: Any, condition: QueryCondition) -> bool:
        """Check if value matches a single condition"""
        if isinstance(value, dict):
            field_value = value.get(condition.field)
        else:
            field_value = getattr(value, condition.field, None)
        
        if field_value is None:
            return condition.operator == QueryOperator.EXISTS and condition.value is False
        
        try:
            if condition.operator == QueryOperator.EQ:
                return field_value == condition.value
            elif condition.operator == QueryOperator.NE:
                return field_value != condition.value
            elif condition.operator == QueryOperator.GT:
                return field_value > condition.value
            elif condition.operator == QueryOperator.GTE:
                return field_value >= condition.value
            elif condition.operator == QueryOperator.LT:
                return field_value < condition.value
            elif condition.operator == QueryOperator.LTE:
                return field_value <= condition.value
            elif condition.operator == QueryOperator.IN:
                return field_value in condition.value
            elif condition.operator == QueryOperator.NIN:
                return field_value not in condition.value
            elif condition.operator == QueryOperator.EXISTS:
                return field_value is not None
            elif condition.operator == QueryOperator.CONTAINS:
                return condition.value in field_value
            elif condition.operator == QueryOperator.STARTS_WITH:
                return field_value.startswith(condition.value)
            elif condition.operator == QueryOperator.ENDS_WITH:
                return field_value.endswith(condition.value)
        except (TypeError, AttributeError):
            return False
        
        return False
    
    def _apply_projection(self, value: Any) -> Any:
        """Apply field projection to result"""
        if isinstance(value, dict):
            return {field: value.get(field) for field in self.projection}
        else:
            return {field: getattr(value, field, None) for field in self.projection}