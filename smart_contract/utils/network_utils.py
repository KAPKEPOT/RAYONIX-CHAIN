# smart_contract/utils/network_utils.py
import socket
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
import re

logger = logging.getLogger("SmartContract.Network")

async def fetch_network_conditions() -> Dict[str, Any]:
    """Fetch current network conditions"""
    try:
        # This would query network peers and services
        # For now, return mock data
        return {
            'latency_ms': 100,
            'throughput_mbps': 100,
            'packet_loss': 0.01,
            'node_count': 50,
            'block_height': 1000000,
            'gas_price': 5,
            'mempool_size': 1000
        }
    except Exception as e:
        logger.error(f"Failed to fetch network conditions: {e}")
        return {}

async def resolve_domain(domain: str) -> Optional[str]:
    """Resolve domain name to IP address"""
    try:
        # Use asyncio for DNS resolution
        loop = asyncio.get_event_loop()
        result = await loop.getaddrinfo(domain, None)
        
        if result:
            return result[0][4][0]  # Return first IP address
        
        return None
    except Exception as e:
        logger.error(f"Domain resolution failed: {e}")
        return None

def validate_ip_address(ip_address: str) -> bool:
    """Validate IP address format"""
    try:
        socket.inet_pton(socket.AF_INET, ip_address)
        return True
    except socket.error:
        try:
            socket.inet_pton(socket.AF_INET6, ip_address)
            return True
        except socket.error:
            return False

async def check_connectivity(host: str, port: int, timeout: int = 5) -> bool:
    """Check network connectivity to host:port"""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception as e:
        logger.debug(f"Connectivity check failed: {e}")
        return False

async def http_request(url: str, method: str = 'GET', 
                      headers: Optional[Dict] = None, 
                      data: Optional[Any] = None,
                      timeout: int = 10) -> Optional[Dict]:
    """Make HTTP request with error handling"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method, url, headers=headers, json=data, timeout=timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"HTTP {method} {url} returned {response.status}")
                    return None
    except Exception as e:
        logger.error(f"HTTP request failed: {e}")
        return None

def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))

def get_network_interfaces() -> Dict[str, Any]:
    """Get network interface information"""
    try:
        import netifaces
        interfaces = {}
        
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            interfaces[interface] = {
                'ipv4': addrs.get(netifaces.AF_INET, []),
                'ipv6': addrs.get(netifaces.AF_INET6, []),
                'mac': addrs.get(netifaces.AF_LINK, [])
            }
        
        return interfaces
    except ImportError:
        logger.warning("netifaces module not available")
        return {}

async measure_latency(host: str, port: int = 80, samples: int = 3) -> float:
    """Measure network latency to host"""
    latencies = []
    
    for _ in range(samples):
        try:
            start_time = asyncio.get_event_loop().time()
            await check_connectivity(host, port, timeout=5)
            end_time = asyncio.get_event_loop().time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        except Exception as e:
            logger.debug(f"Latency measurement failed: {e}")
    
    if latencies:
        return sum(latencies) / len(latencies)
    else:
        return float('inf')