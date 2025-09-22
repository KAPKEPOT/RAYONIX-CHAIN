import asyncio
from typing import Tuple

class UDPProtocol(asyncio.DatagramProtocol):
    """UDP protocol implementation for asyncio"""
    
    def __init__(self, handler):
        self.handler = handler
        self.transport = None
    
    def connection_made(self, transport):
        self.transport = transport
    
    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        asyncio.create_task(self.handler.handle_connection(data, addr))
    
    def error_received(self, exc):
        pass
    
    def connection_lost(self, exc):
        pass