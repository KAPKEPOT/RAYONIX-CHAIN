# database/features/query_builder.py
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
    def __init__(self, db):
        self.db = db
        self.conditions: List[QueryCondition] = []
        self._limit: Optional[int] = None          # CHANGED: self.limit → self._limit
        self._offset: Optional[int] = None         # CHANGED: self.offset → self._offset
        self._sort_field: Optional[str] = None     # CHANGED: self.sort_field → self._sort_field
        self._sort_reverse: bool = False           # CHANGED: self.sort_reverse → self._sort_reverse
        self._projection: Optional[List[str]] = None  # CHANGED: self.projection → self._projection
    
    def where(self, field: str, operator: QueryOperator, value: Any) -> 'QueryBuilder':
        """Add a condition to the query"""
        self.conditions.append(QueryCondition(field, operator, value))
        return self
    
    def limit(self, limit: int) -> 'QueryBuilder':
        """Set result limit"""
        self._limit = limit  # CHANGED: self.limit → self._limit
        return self
    
    def offset(self, offset: int) -> 'QueryBuilder':
        """Set result offset"""
        self._offset = offset  # CHANGED: self.offset → self._offset
        return self
    
    def sort(self, field: str, reverse: bool = False) -> 'QueryBuilder':
        """Set sort field and direction"""
        self._sort_field = field      # CHANGED: self.sort_field → self._sort_field
        self._sort_reverse = reverse  # CHANGED: self.sort_reverse → self._sort_reverse
        return self
    
    def select(self, fields: List[str]) -> 'QueryBuilder':
        """Set projection fields"""
        self._projection = fields  # CHANGED: self.projection → self._projection
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
        for index_name, index in self.db.indexes.items():
            if hasattr(index, 'config') and hasattr(index.config, 'fields'):
                if condition.field in index.config.fields:
                    return True
        return False
    
    def _execute_index_query(self, condition: QueryCondition) -> List[Any]:
        """Execute query using index"""
        # Find the first index that covers this field
        for index_name, index in self.db.indexes.items():
            if hasattr(index, 'config') and hasattr(index.config, 'fields'):
                if condition.field in index.config.fields:
                    if condition.operator == QueryOperator.EQ:
                        return index.query(condition.value, self._limit or 1000, self._offset or 0)  # CHANGED: self.limit → self._limit, self.offset → self._offset
        
        # Fall back to filter if no suitable index found
        return self._execute_filter_query()
    
    def _execute_filter_query(self) -> List[Any]:
        """Execute query by filtering all documents"""
        results = []
        count = 0
        skipped = 0
        
        for key, value in self.db.iterate():
            if self._offset and skipped < self._offset:  # CHANGED: self.offset → self._offset
                skipped += 1
                continue
            
            if self._matches_all_conditions(value):
                if self._projection:  # CHANGED: self.projection → self._projection
                    results.append(self._apply_projection(value))
                else:
                    results.append(value)
                
                count += 1
                if self._limit and count >= self._limit:  # CHANGED: self.limit → self._limit
                    break
        
        if self._sort_field:  # CHANGED: self.sort_field → self._sort_field
            results.sort(
                key=lambda x: x.get(self._sort_field) if isinstance(x, dict) else getattr(x, self._sort_field, None),  # CHANGED: self.sort_field → self._sort_field
                reverse=self._sort_reverse  # CHANGED: self.sort_reverse → self._sort_reverse
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
            return {field: value.get(field) for field in self._projection}  # CHANGED: self.projection → self._projection
        else:
            return {field: getattr(value, field, None) for field in self._projection}  #