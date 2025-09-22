# utils/validators.py - Address validation, etc.

import re
import hashlib
import base58
from typing import Union, Optional

def validate_rayonix_address(address: str) -> bool:
    """
    Validate RAYONIX blockchain address format
    Address format: base58check encoded with version byte and checksum
    """
    if not address or not isinstance(address, str):
        return False
    
    # Check length (typical base58check addresses are 25-35 characters)
    if len(address) < 25 or len(address) > 35:
        return False
    
    # Check character set (base58)
    base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    if not all(c in base58_chars for c in address):
        return False
    
    try:
        # Decode base58
        decoded = base58.b58decode(address)
        
        # Check length (should be 25 bytes: version + payload + checksum)
        if len(decoded) != 25:
            return False
        
        # Extract components
        version = decoded[0]
        payload = decoded[1:-4]
        checksum = decoded[-4:]
        
        # Verify checksum
        calculated_checksum = hashlib.sha256(hashlib.sha256(decoded[:-4]).digest()).digest()[:4]
        if checksum != calculated_checksum:
            return False
        
        # Validate version byte based on network
        network_type = _get_network_type_from_version(version)
        if not network_type:
            return False
        
        return True
        
    except Exception:
        return False

def _get_network_type_from_version(version_byte: int) -> Optional[str]:
    """
    Get network type from address version byte
    """
    # Standard version bytes (these would be defined in the blockchain specification)
    version_map = {
        0x00: 'mainnet',  # Bitcoin mainnet P2PKH
        0x05: 'mainnet',  # Bitcoin mainnet P2SH
        0x6F: 'testnet',  # Bitcoin testnet P2PKH
        0xC4: 'testnet',  # Bitcoin testnet P2SH
        # Add RAYONIX-specific version bytes here
        0x1C: 'mainnet',  # Example RAYONIX mainnet
        0x1D: 'testnet',  # Example RAYONIX testnet
    }
    
    return version_map.get(version_byte)

def validate_transaction_hash(tx_hash: str) -> bool:
    """
    Validate transaction hash format (64 character hex string)
    """
    if not tx_hash or not isinstance(tx_hash, str):
        return False
    
    # Check length
    if len(tx_hash) != 64:
        return False
    
    # Check hex format
    if not re.match(r'^[0-9a-fA-F]{64}$', tx_hash):
        return False
    
    return True

def validate_block_hash(block_hash: str) -> bool:
    """
    Validate block hash format (64 character hex string)
    """
    return validate_transaction_hash(block_hash)  # Same format

def validate_amount(amount: Union[int, float, str]) -> bool:
    """
    Validate amount value (positive number)
    """
    try:
        amount_val = float(amount)
        return amount_val >= 0 and not float('inf') == amount_val
    except (ValueError, TypeError):
        return False

def validate_fee(fee: Union[int, float, str]) -> bool:
    """
    Validate fee value (non-negative number)
    """
    try:
        fee_val = float(fee)
        return fee_val >= 0 and not float('inf') == fee_val
    except (ValueError, TypeError):
        return False

def validate_network_type(network_type: str) -> bool:
    """
    Validate network type string
    """
    return network_type in ['mainnet', 'testnet', 'regtest']

def validate_port(port: Union[int, str]) -> bool:
    """
    Validate network port number
    """
    try:
        port_num = int(port)
        return 1 <= port_num <= 65535
    except (ValueError, TypeError):
        return False

def validate_ip_address(ip: str) -> bool:
    """
    Validate IP address format
    """
    # IPv4 pattern
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    
    # IPv6 pattern (simplified)
    ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$'
    
    if re.match(ipv4_pattern, ip):
        # Validate IPv4 octets
        octets = ip.split('.')
        return all(0 <= int(octet) <= 255 for octet in octets)
    elif re.match(ipv6_pattern, ip):
        # Basic IPv6 validation
        return True
    else:
        return False

def validate_node_address(address: str) -> bool:
    """
    Validate node address format (ip:port)
    """
    if ':' not in address:
        return False
    
    ip, port = address.split(':', 1)
    return validate_ip_address(ip) and validate_port(port)

def validate_mnemonic_phrase(phrase: str) -> bool:
    """
    Validate mnemonic recovery phrase
    """
    if not phrase or not isinstance(phrase, str):
        return False
    
    words = phrase.split()
    # Standard mnemonic phrases are 12, 18, or 24 words
    return len(words) in [12, 18, 24] and all(word.isalpha() for word in words)

def validate_private_key(private_key: str) -> bool:
    """
    Validate private key format (hex encoded)
    """
    if not private_key or not isinstance(private_key, str):
        return False
    
    # Check hex format and length (64 characters for 256-bit key)
    return bool(re.match(r'^[0-9a-fA-F]{64}$', private_key))