# helpers.py - MODIFIED VERSION
import os
import secrets
import hashlib
from typing import Optional, Tuple

# Remove direct imports that cause circular dependency
# from rayonix_wallet.core.wallet import RayonixWallet  # REMOVE
# from rayonix_wallet.core.config import WalletConfig   # REMOVE
# from rayonix_wallet.core.types import WalletType      # REMOVE

def generate_mnemonic(strength: int = 256) -> str:
    """Generate BIP39 mnemonic phrase"""
    from mnemonic import Mnemonic
    mnemo = Mnemonic("english")
    return mnemo.generate(strength=strength)

def validate_mnemonic(mnemonic_phrase: str) -> bool:
    """Validate BIP39 mnemonic phrase"""
    from mnemonic import Mnemonic
    mnemo = Mnemonic("english")
    return mnemo.check(mnemonic_phrase)

def mnemonic_to_seed(mnemonic_phrase: str, passphrase: str = "") -> bytes:
    """Convert mnemonic to seed using BIP39"""
    from mnemonic import Mnemonic
    mnemo = Mnemonic("english")
    return mnemo.to_seed(mnemonic_phrase, passphrase)

def create_hd_wallet(config=None) -> Tuple[str, object]:  # Use object instead of RayonixWallet
    """Create new HD wallet with mnemonic"""
    # Lazy imports to break circular dependency
    from rayonix_wallet.core.config import WalletConfig
    from rayonix_wallet.core.wallet import RayonixWallet
    
    config = config or WalletConfig()
    wallet = RayonixWallet(config)
    
    mnemonic = generate_mnemonic()
    success = wallet.create_from_mnemonic(mnemonic)
    
    if not success:
        raise Exception("Failed to create HD wallet")
    
    return mnemonic, wallet

def create_wallet_from_private_key(private_key: str, config=None) -> object:
    """Create wallet from private key"""
    # Lazy imports to break circular dependency
    from rayonix_wallet.core.config import WalletConfig
    from rayonix_wallet.core.wallet import RayonixWallet
    from rayonix_wallet.core.types import WalletType
    
    config = config or WalletConfig()
    wallet = RayonixWallet(config)
    
    success = wallet.create_from_private_key(private_key, WalletType.NON_HD)
    
    if not success:
        raise Exception("Failed to create wallet from private key")
    
    return wallet

# Rest of the functions remain the same (they don't have circular dependencies)
def generate_random_bytes(length: int = 32) -> bytes:
    """Generate cryptographically secure random bytes"""
    return secrets.token_bytes(length)

def generate_random_hex(length: int = 32) -> str:
    """Generate cryptographically secure random hex string"""
    return secrets.token_hex(length)

def hash_data(data: bytes, algorithm: str = "sha256") -> str:
    """Hash data using specified algorithm"""
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    elif algorithm == "blake2b":
        return hashlib.blake2b(data).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def secure_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks"""
    return secrets.compare_digest(a, b)

def format_amount(amount: int, decimals: int = 8) -> str:
    """Format amount with specified decimal places"""
    whole = amount // (10 ** decimals)
    fractional = amount % (10 ** decimals)
    return f"{whole}.{fractional:0{decimals}d}"

def parse_amount(amount_str: str, decimals: int = 8) -> int:
    """Parse amount string to integer"""
    if '.' in amount_str:
        whole, fractional = amount_str.split('.')
        fractional = fractional.ljust(decimals, '0')[:decimals]
        return int(whole) * (10 ** decimals) + int(fractional)
    else:
        return int(amount_str) * (10 ** decimals)

def calculate_fee(input_count: int, output_count: int, fee_rate: int) -> int:
    """Calculate transaction fee"""
    # Base transaction size
    base_size = 10
    # Input size (approx 150 bytes per input)
    input_size = input_count * 150
    # Output size (approx 34 bytes per output)
    output_size = output_count * 34
    
    total_size = base_size + input_size + output_size
    return total_size * fee_rate

def is_valid_derivation_path(path: str) -> bool:
    """Check if derivation path is valid"""
    if not path.startswith('m/'):
        return False
    
    parts = path.split('/')[1:]
    for part in parts:
        if not part[:-1].isdigit() or (part.endswith("'") and not part[:-1].isdigit()):
            return False
    
    return True

def get_default_data_dir() -> str:
    """Get default data directory for wallet files"""
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ['APPDATA'], 'RayonixWallet')
    elif os.name == 'posix':  # macOS/Linux
        return os.path.join(os.path.expanduser('~'), '.rayonixwallet')
    else:
        return os.path.join(os.path.expanduser('~'), 'RayonixWallet')

def ensure_data_dir(data_dir: Optional[str] = None) -> str:
    """Ensure data directory exists and return path"""
    if data_dir is None:
        data_dir = get_default_data_dir()
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, mode=0o700)
    
    return data_dir

def cleanup_sensitive_data(data: str) -> None:
    """Attempt to securely wipe sensitive data from memory"""
    # This is a best-effort approach since Python doesn't guarantee memory wiping
    import ctypes
    if isinstance(data, str):
        # Convert to mutable buffer
        buffer = ctypes.create_string_buffer(data.encode())
        ctypes.memset(ctypes.addressof(buffer), 0, len(buffer))
    # Let garbage collector handle the rest