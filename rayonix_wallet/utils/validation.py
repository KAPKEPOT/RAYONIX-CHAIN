import re
from typing import Optional
import re
import hashlib  # Add this line
from typing import Optional
from rayonix_wallet.core.exceptions import WalletError

def get_address_type():
    from rayonix_wallet.core.types import AddressType
    return AddressType

def validate_address_format(address: str, address_type, network: str = "mainnet") -> bool:
    """Validate cryptocurrency address format"""
    AddressType = get_address_type()
    
    try:
        if address_type == AddressType.RAYONIX:
            return _validate_rayonix_address(address, network)
        elif address_type == AddressType.P2PKH:
            return _validate_p2pkh_address(address, network)
        elif address_type == AddressType.P2WPKH:
            return _validate_bech32_address(address, network)
        elif address_type == AddressType.ETHEREUM:
            return _validate_ethereum_address(address)
        else:
            return _validate_generic_address(address)
    except:
        return False

def _validate_rayonix_address(address: str, network: str) -> bool:
    """Validate Rayonix address format"""
    if not re.match(r'^[1-9A-HJ-NP-Za-km-z]{25,35}$', address):
        return False
    
    # Add more specific validation for Rayonix addresses
    return True

def _validate_p2pkh_address(address: str, network: str) -> bool:
    """Validate P2PKH address format"""
    if network == "mainnet":
        return address.startswith('1')
    else:
        return address.startswith('m') or address.startswith('n')

def _validate_bech32_address(address: str, network: str) -> bool:
    """Validate Bech32 address format"""
    if network == "mainnet":
        return address.startswith('bc1')
    else:
        return address.startswith('tb1')

def _validate_ethereum_address(address: str) -> bool:
    """Validate Ethereum address format"""
    if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
        return False
    
    # Basic checksum validation
    return _validate_eth_checksum(address)

def _validate_eth_checksum(address: str) -> bool:
    """Validate Ethereum address checksum"""
    address = address[2:].lower()
    hash = hashlib.sha3_256(address.encode()).hexdigest()
    
    for i, char in enumerate(address):
        if char.isalpha():
            if hash[i] >= '8' and char.islower():
                return False
            if hash[i] < '8' and char.isupper():
                return False
    return True

def _validate_generic_address(address: str) -> bool:
    """Generic address validation"""
    # Basic length and character validation
    if len(address) < 26 or len(address) > 95:
        return False
    
    if not re.match(r'^[a-zA-Z0-9]+$', address):
        return False
    
    return True

def validate_private_key(private_key: str, key_type: str = "hex") -> bool:
    """Validate private key format"""
    try:
        if key_type == "hex":
            if private_key.startswith('0x'):
                private_key = private_key[2:]
            
            if len(private_key) != 64:
                return False
            
            # Check if it's a valid hex string
            bytes.fromhex(private_key)
            return True
        
        elif key_type == "wif":
            # Wallet Import Format validation
            return _validate_wif_private_key(private_key)
        
        else:
            return False
            
    except:
        return False

def _validate_wif_private_key(private_key: str) -> bool:
    """Validate WIF private key format"""
    if not re.match(r'^[5KL][1-9A-HJ-NP-Za-km-z]{50,51}$', private_key):
        return False
    
    # Add more specific WIF validation
    return True

def validate_mnemonic(mnemonic_phrase: str) -> bool:
    """Validate BIP39 mnemonic phrase"""
    try:
        from mnemonic import Mnemonic
        mnemo = Mnemonic("english")
        return mnemo.check(mnemonic_phrase)
    except:
        return False

def validate_derivation_path(path: str) -> bool:
    """Validate BIP32/BIP44 derivation path"""
    if not path.startswith('m/'):
        return False
    
    parts = path.split('/')[1:]
    for part in parts:
        if not re.match(r'^\d+(\'?)$', part):
            return False
    
    return True

def validate_amount(amount: int, decimals: int = 8) -> bool:
    """Validate amount format and precision"""
    if amount < 0:
        return False
    
    # Check if amount exceeds maximum precision
    max_value = 10**18  # Arbitrary large number
    if amount > max_value:
        return False
    
    return True

def validate_fee_rate(fee_rate: int) -> bool:
    """Validate transaction fee rate"""
    return 1 <= fee_rate <= 1000  # Reasonable bounds for most blockchains