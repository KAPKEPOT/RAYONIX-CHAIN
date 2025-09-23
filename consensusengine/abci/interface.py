# consensus/abci/interface.py
from typing import Callable, Optional, Dict, Any, List
import logging

logger = logging.getLogger('ABCIInterface')

class ABCIApplication:
    """Application Blockchain Interface for decoupling consensus from execution"""
    
    def __init__(self):
        self.check_tx_fn: Optional[Callable[[str], bool]] = None
        self.deliver_tx_fn: Optional[Callable[[str], bool]] = None
        self.commit_fn: Optional[Callable[[], str]] = None
        self.begin_block_fn: Optional[Callable[[int, str], None]] = None
        self.end_block_fn: Optional[Callable[[int], None]] = None
        self.query_fn: Optional[Callable[[str], Any]] = None
        self.info_fn: Optional[Callable[[], Dict[str, Any]]] = None
        
        # State management
        self.app_hash: str = ""
        self.last_block_height: int = 0
        self.last_block_app_hash: str = ""
    
    def set_check_tx(self, fn: Callable[[str], bool]):
        """Set transaction validation function"""
        self.check_tx_fn = fn
    
    def set_deliver_tx(self, fn: Callable[[str], bool]):
        """Set transaction delivery function"""
        self.deliver_tx_fn = fn
    
    def set_commit(self, fn: Callable[[], str]):
        """Set commit function"""
        self.commit_fn = fn
    
    def set_begin_block(self, fn: Callable[[int, str], None]):
        """Set begin block function"""
        self.begin_block_fn = fn
    
    def set_end_block(self, fn: Callable[[int], None]):
        """Set end block function"""
        self.end_block_fn = fn
    
    def set_query(self, fn: Callable[[str], Any]):
        """Set query function"""
        self.query_fn = fn
    
    def set_info(self, fn: Callable[[], Dict[str, Any]]):
        """Set info function"""
        self.info_fn = fn
    
    def check_tx(self, tx_data: str) -> bool:
        """Validate a transaction"""
        if self.check_tx_fn:
            try:
                return self.check_tx_fn(tx_data)
            except Exception as e:
                logger.error(f"Error in check_tx: {e}")
                return False
        return True
    
    def deliver_tx(self, tx_data: str) -> bool:
        """Deliver a transaction to the application"""
        if self.deliver_tx_fn:
            try:
                return self.deliver_tx_fn(tx_data)
            except Exception as e:
                logger.error(f"Error in deliver_tx: {e}")
                return False
        return True
    
    def commit(self) -> str:
        """Commit application state and return app hash"""
        if self.commit_fn:
            try:
                self.app_hash = self.commit_fn()
                return self.app_hash
            except Exception as e:
                logger.error(f"Error in commit: {e}")
                return ""
        return ""
    
    def begin_block(self, height: int, block_hash: str) -> None:
        """Notify application of block beginning"""
        if self.begin_block_fn:
            try:
                self.begin_block_fn(height, block_hash)
            except Exception as e:
                logger.error(f"Error in begin_block: {e}")
    
    def end_block(self, height: int) -> None:
        """Notify application of block ending"""
        if self.end_block_fn:
            try:
                self.end_block_fn(height)
            except Exception as e:
                logger.error(f"Error in end_block: {e}")
    
    def query(self, path: str) -> Any:
        """Query application state"""
        if self.query_fn:
            try:
                return self.query_fn(path)
            except Exception as e:
                logger.error(f"Error in query: {e}")
        return None
    
    def info(self) -> Dict[str, Any]:
        """Get application information"""
        if self.info_fn:
            try:
                info_data = self.info_fn()
                info_data.update({
                    'last_block_height': self.last_block_height,
                    'last_block_app_hash': self.last_block_app_hash,
                    'app_hash': self.app_hash
                })
                return info_data
            except Exception as e:
                logger.error(f"Error in info: {e}")
        
        return {
            'last_block_height': self.last_block_height,
            'last_block_app_hash': self.last_block_app_hash,
            'app_hash': self.app_hash
        }
    
    def update_block_info(self, height: int, app_hash: str):
        """Update block information after commit"""
        self.last_block_height = height
        self.last_block_app_hash = app_hash

class ABCIHandler:
    """Handler for ABCI message processing"""
    
    def __init__(self, abci_app: ABCIApplication):
        self.abci_app = abci_app
    
    def process_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ABCI request and return response"""
        try:
            if request_type == 'info':
                return self._handle_info(request_data)
            elif request_type == 'check_tx':
                return self._handle_check_tx(request_data)
            elif request_type == 'deliver_tx':
                return self._handle_deliver_tx(request_data)
            elif request_type == 'commit':
                return self._handle_commit(request_data)
            elif request_type == 'begin_block':
                return self._handle_begin_block(request_data)
            elif request_type == 'end_block':
                return self._handle_end_block(request_data)
            elif request_type == 'query':
                return self._handle_query(request_data)
            else:
                return {'error': f'Unknown request type: {request_type}'}
                
        except Exception as e:
            logger.error(f"Error processing ABCI request {request_type}: {e}")
            return {'error': str(e)}
    
    def _handle_info(self, request_data: Dict) -> Dict:
        """Handle info request"""
        info = self.abci_app.info()
        return {
            'data': info,
            'last_block_height': info['last_block_height'],
            'last_block_app_hash': info['last_block_app_hash']
        }
    
    def _handle_check_tx(self, request_data: Dict) -> Dict:
        """Handle check_tx request"""
        tx_data = request_data.get('tx', '')
        is_valid = self.abci_app.check_tx(tx_data)
        return {
            'code': 0 if is_valid else 1,
            'log': 'OK' if is_valid else 'Invalid transaction'
        }
    
    def _handle_deliver_tx(self, request_data: Dict) -> Dict:
        """Handle deliver_tx request"""
        tx_data = request_data.get('tx', '')
        success = self.abci_app.deliver_tx(tx_data)
        return {
            'code': 0 if success else 1,
            'log': 'OK' if success else 'Delivery failed'
        }
    
    def _handle_commit(self, request_data: Dict) -> Dict:
        """Handle commit request"""
        app_hash = self.abci_app.commit()
        return {
            'data': app_hash.encode() if app_hash else b'',
            'log': 'OK'
        }
    
    def _handle_begin_block(self, request_data: Dict) -> Dict:
        """Handle begin_block request"""
        height = request_data.get('height', 0)
        hash = request_data.get('hash', '')
        self.abci_app.begin_block(height, hash)
        return {'log': 'OK'}
    
    def _handle_end_block(self, request_data: Dict) -> Dict:
        """Handle end_block request"""
        height = request_data.get('height', 0)
        self.abci_app.end_block(height)
        return {'log': 'OK'}
    
    def _handle_query(self, request_data: Dict) -> Dict:
        """Handle query request"""
        path = request_data.get('path', '')
        result = self.abci_app.query(path)
        return {
            'code': 0,
            'log': 'OK',
            'value': result
        }