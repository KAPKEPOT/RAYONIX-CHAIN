"""
REST API implementation for consensus system
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
from aiohttp import web, ClientSession
from aiohttp.web import Request, Response, Application
import time

from ..exceptions import ConsensusError
from ..utils import RateLimiter, Backoff
from ..metrics import MetricsCollector

logger = logging.getLogger('consensus.api')

@dataclass
class APIResponse:
    """Standard API response format"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'timestamp': self.timestamp
        }

class RESTAPI:
    """REST API server for consensus system"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 26657, 
                 max_workers: int = 10, timeout: int = 30,
                 cors_allowed_origins: List[str] = None):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.timeout = timeout
        self.cors_allowed_origins = cors_allowed_origins or []
        
        self.app = Application()
        self.runner = None
        self.site = None
        
        self.rate_limiter = RateLimiter(100, 1.0)  # 100 requests per second
        self.backoff = Backoff(base_delay=0.1, max_delay=5.0)
        
        self.metrics: Optional[MetricsCollector] = None
        self.handlers: Dict[str, Callable] = {}
        
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_routes(self) -> None:
        """Setup API routes"""
        # Health endpoints
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/ready', self.ready_check)
        
        # Consensus endpoints
        self.app.router.add_get('/consensus/status', self.get_consensus_status)
        self.app.router.add_get('/consensus/validators', self.get_validators)
        self.app.router.get('/consensus/block/{height}', self.get_block)
        
        # Transaction endpoints
        self.app.router.add_post('/tx/broadcast', self.broadcast_transaction)
        self.app.router.add_get('/tx/{hash}', self.get_transaction)
        
        # Network endpoints
        self.app.router.add_get('/network/peers', self.get_peers)
        self.app.router.add_get('/network/info', self.get_network_info)
        
        # Metrics endpoints
        self.app.router.add_get('/metrics', self.get_metrics)
    
    def _setup_middleware(self) -> None:
        """Setup middleware"""
        # CORS middleware
        async def cors_middleware(app, handler):
            async def middleware_handler(request):
                response = await handler(request)
                origin = request.headers.get('Origin', '')
                if origin in self.cors_allowed_origins:
                    response.headers['Access-Control-Allow-Origin'] = origin
                    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                return response
            return middleware_handler
        
        # Rate limiting middleware
        async def rate_limit_middleware(app, handler):
            async def middleware_handler(request):
                if not self.rate_limiter.acquire():
                    return self._error_response("Rate limit exceeded", 429)
                return await handler(request)
            return middleware_handler
        
        # Metrics middleware
        async def metrics_middleware(app, handler):
            async def middleware_handler(request):
                start_time = time.time()
                try:
                    response = await handler(request)
                    duration = time.time() - start_time
                    if self.metrics:
                        self.metrics.record_message_latency('api_request', duration)
                    return response
                except Exception as e:
                    duration = time.time() - start_time
                    if self.metrics:
                        self.metrics.record_message_latency('api_request', duration)
                    raise
            return middleware_handler
        
        # Apply middleware
        self.app.middlewares.extend([
            cors_middleware,
            rate_limit_middleware,
            metrics_middleware
        ])
    
    async def start(self) -> None:
        """Start API server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            
            logger.info(f"REST API server started on {self.host}:{self.port}")
            
        except Exception as e:
            raise ConsensusError(f"Failed to start API server: {e}")
    
    async def stop(self) -> None:
        """Stop API server"""
        try:
            if self.runner:
                await self.runner.cleanup()
            logger.info("REST API server stopped")
        except Exception as e:
            logger.error(f"Error stopping API server: {e}")
    
    def register_handler(self, endpoint: str, handler: Callable) -> None:
        """Register custom API handler"""
        self.handlers[endpoint] = handler
    
    def set_metrics_collector(self, metrics: MetricsCollector) -> None:
        """Set metrics collector"""
        self.metrics = metrics
    
    # API endpoint handlers
    
    async def health_check(self, request: Request) -> Response:
        """Health check endpoint"""
        try:
            health_data = {
                'status': 'healthy',
                'timestamp': time.time(),
                'version': '1.0.0',
                'services': {
                    'consensus': 'running',
                    'network': 'running',
                    'database': 'running'
                }
            }
            
            # Add metrics if available
            if self.metrics:
                health_data['metrics'] = self.metrics.health_check()
            
            return self._success_response(health_data)
            
        except Exception as e:
            return self._error_response(f"Health check failed: {e}", 500)
    
    async def ready_check(self, request: Request) -> Response:
        """Readiness check endpoint"""
        try:
            # Check if all components are ready
            ready = True
            issues = []
            
            # Add component readiness checks here
            if not ready:
                return self._error_response("Not ready", 503, data={'issues': issues})
            
            return self._success_response({'ready': True})
            
        except Exception as e:
            return self._error_response(f"Readiness check failed: {e}", 500)
    
    async def get_consensus_status(self, request: Request) -> Response:
        """Get consensus status"""
        try:
            # This would get actual consensus state from the engine
            status = {
                'height': 0,
                'round': 0,
                'step': 'unknown',
                'validators_count': 0,
                'peers_count': 0,
                'latest_block_hash': '0' * 64,
                'total_stake': 0
            }
            
            return self._success_response(status)
            
        except Exception as e:
            return self._error_response(f"Failed to get consensus status: {e}", 500)
    
    async def get_validators(self, request: Request) -> Response:
        """Get validator information"""
        try:
            # This would get actual validator data
            validators = [
                {
                    'address': '0x' + '0' * 40,
                    'stake': 1000,
                    'status': 'active',
                    'voting_power': 100,
                    'commission_rate': 0.1
                }
            ]
            
            return self._success_response({'validators': validators})
            
        except Exception as e:
            return self._error_response(f"Failed to get validators: {e}", 500)
    
    async def get_block(self, request: Request) -> Response:
        """Get block by height"""
        try:
            height = int(request.match_info['height'])
            
            # This would get actual block data
            block = {
                'height': height,
                'hash': '0' * 64,
                'timestamp': time.time(),
                'validator': '0x' + '0' * 40,
                'transaction_count': 0,
                'parent_hash': '0' * 64
            }
            
            return self._success_response(block)
            
        except ValueError:
            return self._error_response("Invalid block height", 400)
        except Exception as e:
            return self._error_response(f"Failed to get block: {e}", 500)
    
    async def broadcast_transaction(self, request: Request) -> Response:
        """Broadcast transaction"""
        try:
            data = await request.json()
            transaction = data.get('transaction')
            
            if not transaction:
                return self._error_response("Transaction required", 400)
            
            # This would actually broadcast the transaction
            tx_hash = '0x' + '0' * 64  # Placeholder
            
            return self._success_response({'hash': tx_hash})
            
        except json.JSONDecodeError:
            return self._error_response("Invalid JSON", 400)
        except Exception as e:
            return self._error_response(f"Failed to broadcast transaction: {e}", 500)
    
    async def get_transaction(self, request: Request) -> Response:
        """Get transaction by hash"""
        try:
            tx_hash = request.match_info['hash']
            
            # This would get actual transaction data
            transaction = {
                'hash': tx_hash,
                'status': 'pending',
                'block_height': 0,
                'timestamp': time.time(),
                'sender': '0x' + '0' * 40,
                'receiver': '0x' + '0' * 40,
                'amount': 0
            }
            
            return self._success_response(transaction)
            
        except Exception as e:
            return self._error_response(f"Failed to get transaction: {e}", 500)
    
    async def get_peers(self, request: Request) -> Response:
        """Get network peers"""
        try:
            # This would get actual peer data
            peers = [
                {
                    'id': 'peer-1',
                    'address': '127.0.0.1',
                    'port': 26656,
                    'version': '1.0.0',
                    'connected': True
                }
            ]
            
            return self._success_response({'peers': peers})
            
        except Exception as e:
            return self._error_response(f"Failed to get peers: {e}", 500)
    
    async def get_network_info(self, request: Request) -> Response:
        """Get network information"""
        try:
            info = {
                'listening': True,
                'listeners': [f"{self.host}:{self.port}"],
                'peers_count': 0,
                'outgoing_peers': 0,
                'incoming_peers': 0,
                'bytes_sent': 0,
                'bytes_received': 0
            }
            
            return self._success_response(info)
            
        except Exception as e:
            return self._error_response(f"Failed to get network info: {e}", 500)
    
    async def get_metrics(self, request: Request) -> Response:
        """Get metrics data"""
        try:
            if not self.metrics:
                return self._error_response("Metrics not available", 503)
            
            metrics_data = self.metrics.get_metrics_summary()
            return self._success_response(metrics_data)
            
        except Exception as e:
            return self._error_response(f"Failed to get metrics: {e}", 500)
    
    def _success_response(self, data: Any, status: int = 200) -> Response:
        """Create success response"""
        response_data = APIResponse(success=True, data=data).to_dict()
        return web.json_response(response_data, status=status)
    
    def _error_response(self, error: str, status: int = 500, data: Any = None) -> Response:
        """Create error response"""
        response_data = APIResponse(success=False, error=error, data=data).to_dict()
        return web.json_response(response_data, status=status)
    
    async def make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request to external API"""
        try:
            async with ClientSession() as session:
                async with session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"API request failed: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"API request error: {e}")
            return None

class APIClient:
    """Client for making requests to consensus API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:26657", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = ClientSession()
    
    async def close(self) -> None:
        """Close client session"""
        await self.session.close()
    
    async def health_check(self) -> Optional[Dict]:
        """Check API health"""
        return await self._get('/health')
    
    async def get_consensus_status(self) -> Optional[Dict]:
        """Get consensus status"""
        return await self._get('/consensus/status')
    
    async def get_validators(self) -> Optional[Dict]:
        """Get validators"""
        return await self._get('/consensus/validators')
    
    async def get_block(self, height: int) -> Optional[Dict]:
        """Get block by height"""
        return await self._get(f'/consensus/block/{height}')
    
    async def broadcast_transaction(self, transaction: Dict) -> Optional[Dict]:
        """Broadcast transaction"""
        return await self._post('/tx/broadcast', json={'transaction': transaction})
    
    async def get_peers(self) -> Optional[Dict]:
        """Get network peers"""
        return await self._get('/network/peers')
    
    async def _get(self, endpoint: str) -> Optional[Dict]:
        """Make GET request"""
        try:
            async with self.session.get(
                f"{self.base_url}{endpoint}", 
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"GET request failed: {e}")
            return None
    
    async def _post(self, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make POST request"""
        try:
            async with self.session.post(
                f"{self.base_url}{endpoint}", 
                timeout=self.timeout,
                **kwargs
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"POST request failed: {e}")
            return None