# api/server.py - RayonixAPIServer class

import aiohttp
from aiohttp import web
import logging

from api.rest_routes import setup_rest_routes
from api.jsonrpc_methods import setup_jsonrpc_methods

logger = logging.getLogger("rayonix_node.api")

class RayonixAPIServer:
    """API server for RAYONIX blockchain node with JSON-RPC and REST endpoints"""
    
    def __init__(self, node: 'RayonixNode', host: str = "127.0.0.1", port: int = 8545):
        self.node = node
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # Setup routes and handlers
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all API routes"""
        # Setup JSON-RPC endpoint
        self.app.router.add_post('/jsonrpc', self.handle_jsonrpc)
        
        # Setup REST API routes
        setup_rest_routes(self.app, self.node)
    
    async def handle_jsonrpc(self, request):
        """Handle JSON-RPC requests"""
        from jsonrpcserver import async_dispatch
        
        request_data = await request.text()
        response = await async_dispatch(request_data, context=self.node)
        return web.Response(text=response, content_type="application/json")
    
    async def start(self):
        """Start the API server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            logger.info(f"API server started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    async def stop(self):
        """Stop the API server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("API server stopped")