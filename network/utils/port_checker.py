# network/utils/port_checker.py

import socket
import asyncio
from typing import List

async def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding"""
    try:
        # Try to create a socket and bind to the port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False

async def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if await check_port_available(host, port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")

def get_process_using_port(port: int) -> str:
    """Get the process using a specific port (Linux only)"""
    try:
        import subprocess
        result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        return "Unknown process"
    except Exception:
        return "Unable to determine"