# api/server.py - RayonixAPIServer class using FastAPI

import logging
import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json

from rayonix_node.api.rest_routes import setup_rest_routes
from rayonix_node.api.jsonrpc_methods import setup_jsonrpc_methods

logger = logging.getLogger("rayonix_node.api")

class RayonixAPIServer:
    """API server for RAYONIX blockchain node with JSON-RPC and REST endpoints using FastAPI"""
    
    def __init__(self, node: 'RayonixNode', host: str = "127.0.0.1", port: int = 52557):
        self.node = node
        self.host = host
        self.port = port
        
        # Initialize FastAPI application
        self.app = FastAPI(
            title="RAYONIX Blockchain Node API",
            description="REST and JSON-RPC API for RAYONIX blockchain node",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.server = None
        self.config = None
        
        # Setup middleware and routes
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        """Setup FastAPI middleware"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup all API routes"""
        try:
            # Setup JSON-RPC endpoint
            @self.app.post("/jsonrpc")
            async def handle_jsonrpc(request: Request):
                return await self.handle_jsonrpc_request(request)
            
            # Setup REST API routes
            setup_rest_routes(self.app, self.node)
            
            # Add health check endpoint
            @self.app.get("/health")
            async def health_check():
                return {"status": "healthy", "service": "rayonix-node"}
            
            logger.info("FastAPI routes setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup API routes: {e}")
            raise
    
    async def handle_jsonrpc_request(self, request: Request):
        """Handle JSON-RPC requests with FastAPI"""
        try:
            from jsonrpcserver import async_dispatch
            
            # Get request body
            body = await request.body()
            request_text = body.decode('utf-8')
            
            # Dispatch to JSON-RPC methods
            response = await async_dispatch(request_text, context=self.node)
            
            return JSONResponse(content=json.loads(response))
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in JSON-RPC request: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error",
                        "data": str(e)
                    },
                    "id": None
                }
            )
        except Exception as e:
            logger.error(f"JSON-RPC request handling failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    },
                    "id": None
                }
            )
    
    async def start(self):
        """Start the API server using uvicorn"""
        try:
            # Configure uvicorn
            self.config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False,  # We handle our own logging
            )
            
            # Create server instance
            self.server = uvicorn.Server(self.config)
            
            # Start the server
            logger.info(f"Starting FastAPI server on {self.host}:{self.port}")
            await self.server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start FastAPI server: {e}")
            return False
        
        return True
    
    async def stop(self):
        """Stop the API server gracefully"""
        try:
            if self.server:
                logger.info("Stopping FastAPI server...")
                self.server.should_exit = True
                # Give it a moment to shutdown
                await asyncio.sleep(1)
                logger.info("FastAPI server stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping FastAPI server: {e}")