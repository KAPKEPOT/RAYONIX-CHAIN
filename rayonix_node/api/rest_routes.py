# rayonix_node/api/rest_routes.py 

import logging
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Request, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from rayonix_node.utils.validators import validate_rayonix_address, validate_amount

logger = logging.getLogger("rayonix_node.api")

# Security
security = HTTPBearer()
router = APIRouter(prefix="/api/v1", tags=["blockchain"])

# Constants
MAX_MEMPOOL_SIZE = 10000
MAX_CONTRACT_SIZE = 1024 * 1024  # 1MB
MIN_STAKE_AMOUNT = 1000.0
WALLET_BACKUP_DIR = "wallet_backups"

# Pydantic Models for Request/Response Validation
class TransactionRequest(BaseModel):
    to: str
    amount: float = Field(..., gt=0)
    fee: float = Field(0.0, ge=0)
    data: Optional[Dict[str, Any]] = None

    @validator('to')
    def validate_to_address(cls, v):
        if not validate_rayonix_address(v):
            raise ValueError('Invalid Rayonix address format')
        return v

    @validator('amount')
    def validate_amount(cls, v):
        if not validate_amount(v):  # Changed from validate_transaction_amount to validate_amount
            raise ValueError('Invalid transaction amount')
        return v

class TransactionResponse(BaseModel):
    tx_hash: str
    status: str
    timestamp: int
    block_height: Optional[int] = None

class BalanceResponse(BaseModel):
    balance: float
    available: float
    pending: float
    staked: float
    total: float

class AddressesResponse(BaseModel):
    addresses: List[str]
    default_address: str
    address_count: int

class BlockchainStatusResponse(BaseModel):
    height: int
    difficulty: float
    hashrate: float
    mempool_size: int
    network: str
    syncing: bool
    sync_progress: float
    total_transactions: int
    chain_work: str
    best_block_hash: str

class WalletCreateRequest(BaseModel):
    wallet_type: str = Field("hd", pattern="^(hd|legacy)$")
    password: Optional[str] = None
    mnemonic_length: int = Field(12, ge=12, le=24)

class WalletLoadRequest(BaseModel):
    mnemonic: str
    password: Optional[str] = None
    wallet_type: str = Field("hd", pattern="^(hd|legacy)$")

class WalletImportRequest(BaseModel):
    file_path: str
    password: str
    backup_password: Optional[str] = None

class WalletBackupRequest(BaseModel):
    file_path: str
    password: str
    include_private_keys: bool = True

class StakeRequest(BaseModel):
    amount: float = Field(..., gt=0)
    validator_address: Optional[str] = None

    @validator('amount')
    def validate_stake_amount(cls, v):
        if v < MIN_STAKE_AMOUNT:
            raise ValueError(f'Minimum stake amount is {MIN_STAKE_AMOUNT}')
        return v

class ContractDeployRequest(BaseModel):
    code: str
    name: str = Field(..., min_length=1, max_length=100)
    initial_balance: float = Field(0.0, ge=0)
    constructor_args: List[Any] = []

    @validator('code')
    def validate_code_size(cls, v):
        if len(v.encode('utf-8')) > MAX_CONTRACT_SIZE:
            raise ValueError(f'Contract code exceeds maximum size of {MAX_CONTRACT_SIZE} bytes')
        return v

class ContractCallRequest(BaseModel):
    contract_address: str
    function: str = Field(..., min_length=1, max_length=100)
    args: List[Any] = []
    value: float = Field(0.0, ge=0)
    gas_limit: int = Field(1000000, ge=0)

class ValidatorRegistrationRequest(BaseModel):
    validator_address: str
    commission_rate: float = Field(..., ge=0, le=100)
    website: Optional[str] = None
    description: Optional[str] = None

class SyncStatusResponse(BaseModel):
    syncing: bool
    current_block: int
    target_block: int
    sync_progress: float
    peers_connected: int
    blocks_remaining: int
    estimated_time_remaining: int

class MempoolResponse(BaseModel):
    size: int
    bytes: int
    transactions: List[Dict[str, Any]]
    fee_stats: Dict[str, float]

class StakingInfoResponse(BaseModel):
    enabled: bool
    staking: bool
    total_staked: float
    validator_status: str
    expected_rewards: float
    staking_power: int
    last_stake_time: Optional[int]
    staking_balance: float

class ContractInfoResponse(BaseModel):
    address: str
    name: str
    balance: float
    code_hash: str
    creator: str
    created_at: int
    transaction_count: int

# Dependency to get node instance
def get_node(request: Request):
    return request.app.state.node

# Authentication middleware
async def authenticate_request(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
):
    """Authenticate API requests"""
    node = request.app.state.node
    api_key = node.config_manager.get('api.auth_key')
    
    if api_key and credentials.credentials != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return True

def setup_rest_routes(app, node):
    """Setup REST API routes for FastAPI"""
    
    # Store node in app state for access in endpoints
    app.state.node = node
    
    # Include the router
    app.include_router(router)
    
    logger.info("FastAPI REST routes setup completed successfully")

# Utility Functions
def derive_encryption_key(password: str, salt: bytes) -> bytes:
    """Derive encryption key from password"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

async def encrypt_wallet_data(data: dict, password: str) -> dict:
    """Encrypt wallet data for backup"""
    salt = secrets.token_bytes(16)
    key = derive_encryption_key(password, salt)
    fernet = Fernet(key)
    
    encrypted_data = fernet.encrypt(json.dumps(data).encode())
    
    return {
        'encrypted_data': base64.urlsafe_b64encode(encrypted_data).decode(),
        'salt': base64.urlsafe_b64encode(salt).decode(),
        'version': '1.0'
    }

async def decrypt_wallet_data(encrypted_wallet: dict, password: str) -> dict:
    """Decrypt wallet backup data"""
    try:
        salt = base64.urlsafe_b64decode(encrypted_wallet['salt'])
        encrypted_data = base64.urlsafe_b64decode(encrypted_wallet['encrypted_data'])
        
        key = derive_encryption_key(password, salt)
        fernet = Fernet(key)
        
        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid backup password or corrupted backup file")

# Blockchain Endpoints
@router.get("/blockchain/status", response_model=BlockchainStatusResponse)
async def get_blockchain_status(request: Request):
    """Get comprehensive blockchain status"""
    node = request.app.state.node
    status = await node.rayonix_chain.get_blockchain_status()
    return status

@router.get("/blockchain/block/{block_hash}/merkle-proof/{tx_hash}")
async def get_merkle_proof(block_hash: str, tx_hash: str, format: str = "binary", request: Request = None):                                                 
    """Get Merkle proof for transaction inclusion in block"""
    node = request.app.state.node
    block = node.rayonix_chain.get_block(block_hash)
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    
    # Convert format string to ProofFormat enum
    proof_format = ProofFormat.BINARY
    if format.lower() == "json":
        proof_format = ProofFormat.JSON
    elif format.lower() == "msgpack":
        proof_format = ProofFormat.MSGPACK
    
    proof = block.get_merkle_proof(tx_hash, proof_format)
    if not proof:
        raise HTTPException(status_code=404, detail="Transaction not found in block")
    
    response_data = {
        "block_hash": block_hash,
        "tx_hash": tx_hash,
        "merkle_root": block.header.merkle_root,
        "transaction_count": len(block.transactions),
        "proof_format": format
    }
    
    if proof_format == ProofFormat.BINARY:
        response_data["proof_hex"] = proof.hex()
    else:
        response_data["proof"] = proof.decode('utf-8') if isinstance(proof, bytes) else proof
    
    return response_data

@router.post("/blockchain/verify-merkle-proof")
async def verify_merkle_proof(proof_data: dict, request: Request = None):
    """Verify Merkle proof for transaction inclusion"""
    node = request.app.state.node
    block_hash = proof_data.get('block_hash')
    tx_hash = proof_data.get('tx_hash')
    proof_hex = proof_data.get('proof_hex')
    proof_format_str = proof_data.get('proof_format', 'binary')
    
    if not all([block_hash, tx_hash, proof_hex]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    block = node.rayonix_chain.get_block(block_hash)
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    
    # Convert proof format
    proof_format = ProofFormat.BINARY
    if proof_format_str.lower() == "json":
        proof_format = ProofFormat.JSON
    elif proof_format_str.lower() == "msgpack":
        proof_format = ProofFormat.MSGPACK
    
    # Convert hex proof to bytes if needed
    if proof_format == ProofFormat.BINARY:
        try:
            proof_bytes = bytes.fromhex(proof_hex)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid proof hex format")
    else:
        proof_bytes = proof_hex.encode('utf-8')
    
    is_valid = block.verify_merkle_proof(tx_hash, proof_bytes, proof_format)
    
    return {
        "valid": is_valid,
        "block_hash": block_hash,
        "tx_hash": tx_hash,
        "merkle_root": block.header.merkle_root
    }

@router.get("/blockchain/block/{block_hash}/light-header")
async def get_light_client_header(block_hash: str, request: Request = None):
    """Get light client header information"""
    node = request.app.state.node
    block = node.rayonix_chain.get_block(block_hash)
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    
    return block.get_light_client_header()

@router.get("/blockchain/transaction/{tx_hash}")
async def get_transaction(tx_hash: str, request: Request):
    """Get transaction by hash with detailed information"""
    node = request.app.state.node
    try:
        chain = node.rayonix_chain
        transaction = chain.get_transaction(tx_hash)
        
        if not transaction:
            # Check mempool
            if tx_hash in chain.mempool:
                transaction = chain.mempool[tx_hash]
                transaction['confirmations'] = 0
                transaction['status'] = 'pending'
            else:
                raise HTTPException(status_code=404, detail="Transaction not found")
        else:
            transaction['confirmations'] = chain.get_block_count() - transaction.get('block_height', 0) + 1
            transaction['status'] = 'confirmed'
        
        # Add input/output details
        if 'inputs' in transaction:
            transaction['input_amount'] = sum(inp.get('amount', 0) for inp in transaction['inputs'])
        if 'outputs' in transaction:
            transaction['output_amount'] = sum(out.get('amount', 0) for out in transaction['outputs'])
            transaction['fee'] = transaction.get('input_amount', 0) - transaction['output_amount']
        
        return transaction
    except Exception as e:
        logger.error(f"Error getting transaction {tx_hash}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/blockchain/blocks")
async def get_blocks(
    request: Request,
    start_height: int = Query(0, ge=0),
    count: int = Query(10, ge=1, le=100)
):
    """Get multiple blocks with pagination"""
    node = request.app.state.node
    try:
        chain = node.rayonix_chain
        current_height = chain.get_block_count()
        
        if start_height > current_height:
            return {"blocks": [], "total_blocks": current_height + 1}
        
        blocks = []
        for height in range(start_height, min(start_height + count, current_height + 1)):
            block = chain.get_block_by_height(height)
            if block:
                blocks.append(block)
        
        return {
            "blocks": blocks,
            "start_height": start_height,
            "end_height": start_height + len(blocks) - 1,
            "total_blocks": current_height + 1,
            "has_more": start_height + len(blocks) <= current_height
        }
    except Exception as e:
        logger.error(f"Error getting blocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Wallet Endpoints
@router.post("/wallet/create", response_model=Dict[str, Any])
async def create_wallet(
    wallet_data: WalletCreateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    print("=== DEBUG: REST /wallet/create ENDPOINT CALLED ===")
    node = request.app.state.node
    
    print(f"DEBUG: Node wallet exists: {node.wallet is not None}")
    
    if node.wallet:
        print("DEBUG: ❌ Wallet already exists, returning error")
        raise HTTPException(status_code=400, detail="Wallet already exists")
    
    try:
        print("DEBUG: Calling node._create_wallet_on_demand()...")
        success = await node._create_wallet_on_demand()
        print(f"DEBUG: _create_wallet_on_demand result: {success}")
        print(f"DEBUG: Node wallet after creation: {node.wallet is not None}")
        
        if not success or not node.wallet:
            print("DEBUG: ❌ Wallet creation failed")
            raise HTTPException(status_code=500, detail="Failed to create wallet")
        
        wallet = node.wallet
        print(f"DEBUG: Wallet ID: {wallet.wallet_id}")
        print(f"DEBUG: Wallet attributes: {dir(wallet)}")
        
        # Get wallet ID safely
        wallet_id = getattr(wallet, 'wallet_id', 'unknown')
        print(f"DEBUG: Wallet ID: {wallet_id}")
        
        # Get addresses safely
        addresses_dict = {}
        if hasattr(wallet, 'get_addresses') and callable(wallet.get_addresses):
        	addresses_dict = wallet.get_addresses()
        elif hasattr(wallet, 'addresses'):
        	addresses_dict = wallet.addresses
        
        first_address = "NO_ADDRESS_GENERATED"
        if addresses_dict:
        	address_keys = list(addresses_dict.keys())
        	if address_keys:
        		first_address = address_keys[0]
        	
        # Get the mnemonic from wallet creation
        mnemonic = None
        if hasattr(wallet, 'creation_mnemonic') and wallet.creation_mnemonic:
        	mnemonic = wallet.creation_mnemonic
        elif hasattr(wallet, '_creation_mnemonic'):
        	# If encrypted, we need to handle this properly
        	mnemonic = "ENCRYPTED_MNEMONIC_NEEDS_DECRYPTION"
        else:
        	mnemonic = "MNEMONIC_NOT_AVAILABLE"
        
        response = {
            "wallet_id": wallet_id,
            "address": first_address,
            "wallet_type": "hd",
            "encrypted": False,
            "mnemonic": mnemonic,
            "address_count": len(addresses_dict)
        }
        
        print("DEBUG: ✅ Wallet creation response prepared")
        return response
        
    except Exception as e:
        print(f"DEBUG: ❌ ENDPOINT ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wallet/load", response_model=Dict[str, Any])
async def load_wallet(
    wallet_data: WalletLoadRequest,
    request: Request,
    #auth: bool = Depends(authenticate_request)
):
    """Load wallet from mnemonic phrase"""
    node = request.app.state.node
    try:
        if node.wallet:
            raise HTTPException(status_code=400, detail="Wallet already loaded")
            
        # Use the new node method
        success = await node.load_wallet_on_demand(
            wallet_data.mnemonic,
            wallet_data.password
        )
        
        if not success:
        	raise HTTPException(status_code=400, detail="Failed to load wallet")
        
        wallet_info = node.wallet.get_wallet_info()
        addresses = node.wallet.get_addresses()
        
        response = {
            "wallet_id": wallet_info.get('wallet_id', 'unknown'),
            "addresses": list(addresses.keys()),
            "address_count": len(addresses),
            "wallet_type": wallet_info.get('wallet_type'),
            "balance": wallet.get_balance(),
            "loaded": True
        }
        
        logger.info(f"Wallet loaded: {response['wallet_id']}")
        return response
        
    except Exception as e:
        logger.error(f"Error loading wallet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wallet/import", response_model=Dict[str, Any])
async def import_wallet(
    wallet_data: WalletImportRequest,
    request: Request,
    #auth: bool = Depends(authenticate_request)
):
    """Import wallet from backup file"""
    node = request.app.state.node
    try:
        if node.wallet:
            raise HTTPException(status_code=400, detail="Wallet already loaded")
        
        backup_file = Path(wallet_data.file_path)
        if not backup_file.exists():
            raise HTTPException(status_code=404, detail="Backup file not found")
        
        async with aiofiles.open(backup_file, 'r') as f:
            backup_content = await f.read()
            backup_data = json.loads(backup_content)
        
        # Decrypt if encrypted
        if 'encrypted_data' in backup_data:
            if not wallet_data.backup_password:
                raise HTTPException(status_code=400, detail="Backup password required")
            backup_data = await decrypt_wallet_data(backup_data, wallet_data.backup_password)
        
        # Create wallet from backup data
        wallet_type = backup_data.get('wallet_type', 'hd')
        if wallet_type == "hd":
            wallet = HDWallet()
            if 'mnemonic' not in backup_data:
                raise HTTPException(status_code=400, detail="Invalid backup file: mnemonic missing")
            wallet.create_from_mnemonic(backup_data['mnemonic'])
        else:
            wallet = LegacyWallet()
            if 'private_key' not in backup_data:
                raise HTTPException(status_code=400, detail="Invalid backup file: private key missing")
            wallet.import_from_private_key(backup_data['private_key'])
        
        # Encrypt wallet if password provided
        if wallet_data.password:
            wallet.encrypt(wallet_data.password)
        
        node.wallet = wallet
        
        # Scan for transactions
        await node.scan_wallet_transactions()
        
        response = {
            "wallet_id": wallet.get_wallet_id(),
            "addresses": wallet.get_addresses(),
            "balance": wallet.get_balance(),
            "wallet_type": wallet_type,
            "encrypted": wallet_data.password is not None,
            "imported_from": str(backup_file)
        }
        
        logger.info(f"Wallet imported from backup: {response['wallet_id']}")
        return response
        
    except Exception as e:
        logger.error(f"Error importing wallet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wallet/backup", response_model=Dict[str, Any])
async def backup_wallet(
    backup_data: WalletBackupRequest,
    request: Request,
    #auth: bool = Depends(authenticate_request)
):
    """Backup wallet to encrypted file"""
    node = request.app.state.node
    try:
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        wallet = node.wallet
        
        # Prepare backup data
        backup_info = {
            'wallet_type': wallet.get_wallet_type(),
            'addresses': wallet.get_addresses(),
            'public_keys': wallet.get_public_keys(),
            'created_at': int(datetime.now().timestamp()),
            'network': node.config_manager.get('network.network_type', 'testnet'),
            'version': '1.0'
        }
        
        # Include sensitive data if requested and wallet is not encrypted
        if backup_data.include_private_keys and not wallet.is_encrypted():
            if hasattr(wallet, 'get_mnemonic'):
                backup_info['mnemonic'] = wallet.get_mnemonic()
            else:
                backup_info['private_key'] = wallet.export_private_key()
        
        # Encrypt backup
        encrypted_backup = await encrypt_wallet_data(backup_info, backup_data.password)
        
        # Write to file
        backup_file = Path(backup_data.file_path)
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(backup_file, 'w') as f:
            await f.write(json.dumps(encrypted_backup, indent=2))
        
        response = {
            "backup_file": str(backup_file),
            "encrypted": True,
            "size": backup_file.stat().st_size if backup_file.exists() else 0,
            "timestamp": int(datetime.now().timestamp())
        }
        
        logger.info(f"Wallet backed up to: {backup_file}")
        return response
        
    except Exception as e:
        logger.error(f"Error backing up wallet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/balance", response_model=BalanceResponse)
async def get_wallet_balance(request: Request):
    """Get comprehensive wallet balance information"""
    node = request.app.state.node
    try:
        if not node.wallet:
            return {
                "balance": 0,
                "available": 0,
                "pending": 0,
                "staked": 0,
                "total": 0
            }
        
        wallet = node.wallet
        chain = node.rayonix_chain
        
        # Calculate different balance types
        available_balance = wallet.get_balance()
        pending_balance = wallet.get_pending_balance()
        staked_balance = 0
        
        # Calculate staked balance if staking is enabled
        if hasattr(node, 'staking_manager') and node.staking_manager:
            staked_balance = node.staking_manager.get_staked_balance(wallet.get_addresses())
        
        total_balance = available_balance + pending_balance + staked_balance
        
        return {
            "balance": available_balance,
            "available": available_balance,
            "pending": pending_balance,
            "staked": staked_balance,
            "total": total_balance
        }
    except Exception as e:
        logger.error(f"Error getting wallet balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/addresses", response_model=AddressesResponse)
async def get_wallet_addresses(request: Request):
    """Get all wallet addresses with metadata"""
    node = request.app.state.node
    try:
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        addresses = node.wallet.get_addresses()
        default_address = addresses[0] if addresses else ""
        
        return {
            "addresses": addresses,
            "default_address": default_address,
            "address_count": len(addresses)
        }
    except Exception as e:
        logger.error(f"Error getting wallet addresses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/info", response_model=Dict[str, Any])
async def get_wallet_info(request: Request):
    """Get detailed wallet information"""
    node = request.app.state.node
    try:
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        wallet = node.wallet
        addresses = wallet.get_addresses()
        
        info = {
            "type": wallet.get_wallet_type(),
            "encrypted": wallet.is_encrypted(),
            "address_count": len(addresses),
            "backup_created": wallet.is_backup_created(),
            "wallet_id": wallet.get_wallet_id(),
            "created_at": wallet.get_creation_time(),
            "last_used": wallet.get_last_used_time(),
            "default_address": addresses[0] if addresses else "",
            "balance_breakdown": await get_wallet_balance(request)
        }
        
        # Add HD wallet specific info
        if hasattr(wallet, 'get_account_index'):
            info['account_index'] = wallet.get_account_index()
            info['gap_limit'] = wallet.get_gap_limit()
        
        return info
    except Exception as e:
        logger.error(f"Error getting wallet info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wallet/send", response_model=TransactionResponse)
async def send_transaction(
    transaction: TransactionRequest,
    request: Request,
    auth: bool = Depends(authenticate_request)
):
    """Send transaction with comprehensive validation"""
    node = request.app.state.node
    try:
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        wallet = node.wallet
        chain = node.rayonix_chain
        
        # Validate available balance
        available_balance = wallet.get_balance()
        total_amount = transaction.amount + transaction.fee
        
        if available_balance < total_amount:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient balance. Available: {available_balance}, Required: {total_amount}"
            )
        
        # Create transaction
        tx_data = {
            'version': 1,
            'inputs': [],
            'outputs': [
                {
                    'address': transaction.to,
                    'amount': transaction.amount
                }
            ],
            'timestamp': int(datetime.now().timestamp()),
            'data': transaction.data or {}
        }
        
        # Add change output if needed
        if transaction.fee > 0:
            change = available_balance - total_amount
            if change > 0:
                change_address = wallet.get_addresses()[0]
                tx_data['outputs'].append({
                    'address': change_address,
                    'amount': change
                })
        
        # Sign transaction
        signed_tx = wallet.sign_transaction(tx_data)
        
        # Validate transaction
        if not chain.validate_transaction(signed_tx):
            raise HTTPException(status_code=400, detail="Transaction validation failed")
        
        # Broadcast transaction
        tx_hash = await node.broadcast_transaction(signed_tx)
        
        if not tx_hash:
            raise HTTPException(status_code=500, detail="Failed to broadcast transaction")
        
        # Add to local mempool
        chain.add_to_mempool(signed_tx)
        
        response = {
            "tx_hash": tx_hash,
            "status": "pending",
            "timestamp": signed_tx['timestamp'],
            "block_height": None
        }
        
        logger.info(f"Transaction sent: {tx_hash}, Amount: {transaction.amount}")
        return response
        
    except Exception as e:
        logger.error(f"Error sending transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/transactions")
async def get_transaction_history(
    request: Request,
    count: int = Query(50, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    address: Optional[str] = None
):
    """Get transaction history with pagination and filtering"""
    node = request.app.state.node
    try:
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        wallet = node.wallet
        chain = node.rayonix_chain
        
        # Get transactions from wallet history
        transactions = wallet.get_transaction_history(count, skip)
        
        # Enhance transaction data
        enhanced_transactions = []
        for tx in transactions:
            enhanced_tx = tx.copy()
            
            # Add confirmation count and status
            if 'block_height' in tx:
                confirmations = chain.get_block_count() - tx['block_height'] + 1
                enhanced_tx['confirmations'] = confirmations
                enhanced_tx['status'] = 'confirmed'
            else:
                enhanced_tx['confirmations'] = 0
                enhanced_tx['status'] = 'pending'
            
            # Calculate amounts
            if 'inputs' in tx:
                enhanced_tx['input_amount'] = sum(inp.get('amount', 0) for inp in tx['inputs'])
            if 'outputs' in tx:
                enhanced_tx['output_amount'] = sum(out.get('amount', 0) for out in tx['outputs'])
                enhanced_tx['fee'] = enhanced_tx.get('input_amount', 0) - enhanced_tx['output_amount']
            
            enhanced_transactions.append(enhanced_tx)
        
        # Filter by address if specified
        if address:
            enhanced_transactions = [
                tx for tx in enhanced_transactions
                if any(
                    address in inp.get('address', '') or address in out.get('address', '')
                    for inp in tx.get('inputs', [])
                    for out in tx.get('outputs', [])
                )
            ]
        
        return {
            "transactions": enhanced_transactions,
            "total_count": len(enhanced_transactions),
            "has_more": len(enhanced_transactions) == count
        }
    except Exception as e:
        logger.error(f"Error getting transaction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Staking Endpoints
@router.get("/staking/info", response_model=StakingInfoResponse)
async def get_staking_info(request: Request):
    """Get comprehensive staking information"""
    node = request.app.state.node
    try:
        if not hasattr(node, 'staking_manager') or not node.staking_manager:
            return {
                "enabled": False,
                "staking": False,
                "total_staked": 0,
                "validator_status": "disabled",
                "expected_rewards": 0,
                "staking_power": 0,
                "last_stake_time": None,
                "staking_balance": 0
            }
        
        staking_manager = node.staking_manager
        
        info = {
            "enabled": staking_manager.is_staking_enabled(),
            "staking": staking_manager.is_staking_active(),
            "total_staked": staking_manager.get_total_staked(),
            "validator_status": staking_manager.get_validator_status(),
            "expected_rewards": staking_manager.calculate_expected_rewards(),
            "staking_power": staking_manager.get_staking_power(),
            "last_stake_time": staking_manager.get_last_stake_time(),
            "staking_balance": staking_manager.get_staking_balance()
        }
        
        return info
    except Exception as e:
        logger.error(f"Error getting staking info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/staking/stake", response_model=Dict[str, Any])
async def stake_tokens(
    stake_data: StakeRequest,
    request: Request,
    auth: bool = Depends(authenticate_request)
):
    """Stake tokens for validation with comprehensive validation"""
    node = request.app.state.node
    try:
        if not hasattr(node, 'staking_manager') or not node.staking_manager:
            raise HTTPException(status_code=400, detail="Staking not enabled")
        
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        staking_manager = node.staking_manager
        wallet = node.wallet
        
        # Validate stake amount
        available_balance = wallet.get_balance()
        if available_balance < stake_data.amount:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient balance. Available: {available_balance}, Required: {stake_data.amount}"
            )
        
        if stake_data.amount < MIN_STAKE_AMOUNT:
            raise HTTPException(
                status_code=400,
                detail=f"Minimum stake amount is {MIN_STAKE_AMOUNT}"
            )
        
        # Create stake transaction
        stake_tx = await staking_manager.create_stake_transaction(
            wallet,
            stake_data.amount,
            stake_data.validator_address
        )
        
        if not stake_tx:
            raise HTTPException(status_code=400, detail="Failed to create stake transaction")
        
        # Sign and broadcast transaction
        signed_tx = wallet.sign_transaction(stake_tx)
        tx_hash = await node.broadcast_transaction(signed_tx)
        
        if not tx_hash:
            raise HTTPException(status_code=500, detail="Failed to broadcast stake transaction")
        
        # Register stake in staking manager
        await staking_manager.register_stake(
            wallet.get_addresses()[0],
            stake_data.amount,
            tx_hash
        )
        
        response = {
            "staked_amount": stake_data.amount,
            "validator_address": stake_data.validator_address or wallet.get_addresses()[0],
            "staking_power": staking_manager.calculate_staking_power(stake_data.amount),
            "transaction_hash": tx_hash,
            "expected_rewards": staking_manager.calculate_expected_rewards_for_stake(stake_data.amount)
        }
        
        logger.info(f"Tokens staked: {stake_data.amount}, TX: {tx_hash}")
        return response
        
    except Exception as e:
        logger.error(f"Error staking tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/validators/list", response_model=List[Dict[str, Any]])
async def get_validators(
    request: Request,
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get validator list with comprehensive information"""
    node = request.app.state.node
    try:
        if not hasattr(node, 'consensus_manager') or not node.consensus_manager:
            return []
        
        consensus_manager = node.consensus_manager
        validators = await consensus_manager.get_validators(active_only, limit)
        
        # Enhance validator data
        enhanced_validators = []
        for validator in validators:
            enhanced_validator = validator.copy()
            
            # Calculate additional metrics
            if 'stake' in validator:
                enhanced_validator['staking_power'] = consensus_manager.calculate_staking_power(validator['stake'])
                enhanced_validator['expected_rewards'] = consensus_manager.calculate_validator_rewards(validator['stake'])
                enhanced_validator['uptime'] = consensus_manager.get_validator_uptime(validator['address'])
                enhanced_validator['performance'] = consensus_manager.get_validator_performance(validator['address'])
            
            enhanced_validators.append(enhanced_validator)
        
        return enhanced_validators
    except Exception as e:
        logger.error(f"Error getting validators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validators/register", response_model=Dict[str, Any])
async def register_validator(
    validator_data: ValidatorRegistrationRequest,
    request: Request,
    auth: bool = Depends(authenticate_request)
):
    """Register as a validator"""
    node = request.app.state.node
    try:
        if not hasattr(node, 'consensus_manager') or not node.consensus_manager:
            raise HTTPException(status_code=400, detail="Consensus manager not available")
        
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        consensus_manager = node.consensus_manager
        wallet = node.wallet
        
        # Validate minimum stake
        min_stake = consensus_manager.get_minimum_stake()
        available_balance = wallet.get_balance()
        
        if available_balance < min_stake:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient balance for validator registration. Minimum: {min_stake}"
            )
        
        # Register validator
        result = await consensus_manager.register_validator(
            wallet,
            validator_data.validator_address,
            validator_data.commission_rate,
            validator_data.website,
            validator_data.description
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Registration failed'))
        
        response = {
            "validator_address": validator_data.validator_address,
            "commission_rate": validator_data.commission_rate,
            "registration_tx": result.get('transaction_hash'),
            "status": "pending",
            "min_stake": min_stake
        }
        
        logger.info(f"Validator registered: {validator_data.validator_address}")
        return response
        
    except Exception as e:
        logger.error(f"Error registering validator: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Smart Contract Endpoints
@router.get("/contracts", response_model=List[ContractInfoResponse])
async def get_contracts(
    request: Request,
    deployed_by: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get deployed smart contracts with filtering"""
    node = request.app.state.node
    try:
        if not hasattr(node, 'contract_manager') or not node.contract_manager:
            return []
        
        contract_manager = node.contract_manager
        contracts = await contract_manager.get_contracts(deployed_by, limit)
        
        enhanced_contracts = []
        for contract in contracts:
            enhanced_contract = ContractInfoResponse(
                address=contract['address'],
                name=contract.get('name', 'Unnamed'),
                balance=contract.get('balance', 0),
                code_hash=contract.get('code_hash', ''),
                creator=contract.get('creator', ''),
                created_at=contract.get('created_at', 0),
                transaction_count=contract.get('transaction_count', 0)
            )
            enhanced_contracts.append(enhanced_contract.dict())
        
        return enhanced_contracts
    except Exception as e:
        logger.error(f"Error getting contracts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/contracts/deploy", response_model=Dict[str, Any])
async def deploy_contract(
    contract_data: ContractDeployRequest,
    request: Request,
    auth: bool = Depends(authenticate_request)
):
    """Deploy smart contract with comprehensive validation"""
    node = request.app.state.node
    try:
        if not hasattr(node, 'contract_manager') or not node.contract_manager:
            raise HTTPException(status_code=400, detail="Smart contracts not enabled")
        
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        contract_manager = node.contract_manager
        wallet = node.wallet
        
        # Validate contract code
        validation_result = await contract_manager.validate_contract_code(contract_data.code)
        if not validation_result.get('valid'):
            raise HTTPException(
                status_code=400,
                detail=f"Contract validation failed: {validation_result.get('error')}"
            )
        
        # Check deployment cost
        deployment_cost = await contract_manager.calculate_deployment_cost(contract_data.code)
        total_cost = deployment_cost + contract_data.initial_balance
        
        available_balance = wallet.get_balance()
        if available_balance < total_cost:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient balance for contract deployment. Required: {total_cost}, Available: {available_balance}"
            )
        
        # Deploy contract
        result = await contract_manager.deploy_contract(
            wallet,
            contract_data.code,
            contract_data.name,
            contract_data.initial_balance,
            contract_data.constructor_args
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Deployment failed'))
        
        response = {
            "contract_address": result['contract_address'],
            "transaction_hash": result['transaction_hash'],
            "gas_used": result.get('gas_used', 0),
            "deployment_cost": deployment_cost,
            "initial_balance": contract_data.initial_balance,
            "code_size": len(contract_data.code.encode('utf-8')),
            "code_hash": result.get('code_hash', '')
        }
        
        logger.info(f"Contract deployed: {result['contract_address']}, Name: {contract_data.name}")
        return response
        
    except Exception as e:
        logger.error(f"Error deploying contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/contracts/call", response_model=Dict[str, Any])
async def call_contract(
    call_data: ContractCallRequest,
    request: Request,
    auth: bool = Depends(authenticate_request)
):
    """Call contract function with gas estimation and validation"""
    node = request.app.state.node
    try:
        if not hasattr(node, 'contract_manager') or not node.contract_manager:
            raise HTTPException(status_code=400, detail="Smart contracts not enabled")
        
        if not node.wallet:
            raise HTTPException(status_code=400, detail="No wallet available")
        
        contract_manager = node.contract_manager
        wallet = node.wallet
        
        # Validate contract existence
        contract_exists = await contract_manager.contract_exists(call_data.contract_address)
        if not contract_exists:
            raise HTTPException(status_code=404, detail="Contract not found")
        
        # Estimate gas
        gas_estimate = await contract_manager.estimate_gas(
            call_data.contract_address,
            call_data.function,
            call_data.args
        )
        
        if gas_estimate > call_data.gas_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Gas estimate ({gas_estimate}) exceeds gas limit ({call_data.gas_limit})"
            )
        
        # Check if function exists and is callable
        function_info = await contract_manager.get_function_info(
            call_data.contract_address,
            call_data.function
        )
        
        if not function_info:
            raise HTTPException(status_code=400, detail=f"Function {call_data.function} not found")
        
        if function_info.get('payable', False) and call_data.value == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Function {call_data.function} is payable but no value provided"
            )
        
        # Check balance for value transfer
        if call_data.value > 0:
            available_balance = wallet.get_balance()
            if available_balance < call_data.value:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient balance for value transfer. Required: {call_data.value}, Available: {available_balance}"
                )
        
        # Call contract
        result = await contract_manager.call_contract(
            wallet,
            call_data.contract_address,
            call_data.function,
            call_data.args,
            call_data.value,
            call_data.gas_limit
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Contract call failed'))
        
        response = {
            "result": result.get('return_value'),
            "transaction_hash": result.get('transaction_hash'),
            "gas_used": result.get('gas_used', 0),
            "events": result.get('events', []),
            "status": result.get('status', 'executed')
        }
        
        logger.info(f"Contract called: {call_data.contract_address}.{call_data.function}")
        return response
        
    except Exception as e:
        logger.error(f"Error calling contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/contracts/{contract_address}")
async def get_contract_info(contract_address: str, request: Request):
    """Get detailed contract information"""
    node = request.app.state.node
    try:
        if not hasattr(node, 'contract_manager') or not node.contract_manager:
            raise HTTPException(status_code=400, detail="Smart contracts not enabled")
        
        contract_manager = node.contract_manager
        
        # Get contract details
        contract_info = await contract_manager.get_contract_info(contract_address)
        if not contract_info:
            raise HTTPException(status_code=404, detail="Contract not found")
        
        # Get contract state
        contract_state = await contract_manager.get_contract_state(contract_address)
        
        # Get contract transactions
        contract_txs = await contract_manager.get_contract_transactions(contract_address, 100)
        
        response = {
            **contract_info,
            "state": contract_state,
            "recent_transactions": contract_txs,
            "balance": await contract_manager.get_contract_balance(contract_address),
            "code_size": len(contract_info.get('code', '').encode('utf-8')),
            "active": contract_info.get('active', True)
        }
        
        return response
    except Exception as e:
        logger.error(f"Error getting contract info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Network and System Endpoints
@router.get("/node/status")
async def get_node_status(request: Request):
    """Get comprehensive node status"""
    node = request.app.state.node
    try:
        status = node.state_manager.get_state_summary()
        
        # Add additional node information
        status.update({
            "version": "1.0.0",
            "network": node.config_manager.get('network.network_type', 'testnet'),
            "wallet_loaded": node.wallet is not None,
            "api_enabled": node.api_server is not None,
            "network_enabled": node.network is not None,
            "staking_enabled": hasattr(node, 'staking_manager') and node.staking_manager is not None,
            "contracts_enabled": hasattr(node, 'contract_manager') and node.contract_manager is not None,
            "uptime": node.state_manager.get_uptime(),
            "memory_usage": node.state_manager.get_memory_usage(),
            "cpu_usage": node.state_manager.get_cpu_usage()
        })
        
        return status
    except Exception as e:
        logger.error(f"Error getting node status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/node/peers")
async def get_peers(
    request: Request,
    connected_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get peer information with filtering"""
    node = request.app.state.node
    try:
        if not node.network:
            raise HTTPException(status_code=400, detail="Network not available")
        
        peers = await node.network.get_peers(connected_only, limit)
        
        # Enhance peer data
        enhanced_peers = []
        for peer in peers:
            enhanced_peer = peer.copy()
            
            # Calculate connection metrics
            if 'connected_since' in peer:
                connected_time = datetime.now().timestamp() - peer['connected_since']
                enhanced_peer['connection_duration'] = connected_time
                enhanced_peer['connection_stability'] = node.network.get_connection_stability(peer['id'])
            
            enhanced_peers.append(enhanced_peer)
        
        return {
            "peers": enhanced_peers,
            "total_peers": len(enhanced_peers),
            "connected_peers": len([p for p in enhanced_peers if p.get('connected', False)]),
            "network_id": node.network.get_network_id()
        }
    except Exception as e:
        logger.error(f"Error getting peers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sync/status", response_model=SyncStatusResponse)
async def get_sync_status(request: Request):
    """Get synchronization status with progress information"""
    node = request.app.state.node
    try:
        chain = node.rayonix_chain
        sync_state = chain.get_sync_state()
        
        # Calculate estimated time remaining
        blocks_remaining = sync_state.get('target_block', 0) - sync_state.get('current_block', 0)
        sync_speed = chain.get_sync_speed()  # blocks per second
        
        estimated_time_remaining = 0
        if sync_speed > 0 and blocks_remaining > 0:
            estimated_time_remaining = blocks_remaining / sync_speed
        
        response = SyncStatusResponse(
            syncing=sync_state.get('syncing', False),
            current_block=sync_state.get('current_block', 0),
            target_block=sync_state.get('target_block', 0),
            sync_progress=sync_state.get('sync_progress', 0),
            peers_connected=sync_state.get('peers_connected', 0),
            blocks_remaining=blocks_remaining,
            estimated_time_remaining=int(estimated_time_remaining)
        )
        
        return response.dict()
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mempool", response_model=MempoolResponse)
async def get_mempool_info(
    request: Request,
    include_transactions: bool = Query(False)
):
    """Get mempool information with statistics"""
    node = request.app.state.node
    try:
        chain = node.rayonix_chain
        mempool = chain.mempool
        
        # Calculate fee statistics
        fees = [tx.get('fee', 0) for tx in mempool.values() if 'fee' in tx]
        fee_stats = {
            "min_fee": min(fees) if fees else 0,
            "max_fee": max(fees) if fees else 0,
            "avg_fee": sum(fees) / len(fees) if fees else 0,
            "median_fee": sorted(fees)[len(fees) // 2] if fees else 0,
            "total_fees": sum(fees)
        }
        
        # Prepare transactions for response
        transactions = []
        if include_transactions:
            transactions = list(mempool.values())[:100]  # Limit to first 100
        
        response = MempoolResponse(
            size=len(mempool),
            bytes=sum(len(json.dumps(tx).encode('utf-8')) for tx in mempool.values()),
            transactions=transactions,
            fee_stats=fee_stats
        )
        
        return response.dict()
    except Exception as e:
        logger.error(f"Error getting mempool info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/network/stats")
async def get_network_stats(request: Request):
    """Get comprehensive network statistics"""
    node = request.app.state.node
    try:
        if not node.network:
            raise HTTPException(status_code=400, detail="Network not available")
        
        network_stats = await node.network.get_network_stats()
        
        # Add blockchain statistics
        chain = node.rayonix_chain
        network_stats.update({
            "block_height": chain.get_block_count(),
            "difficulty": chain.get_difficulty(),
            "hashrate": chain.get_network_hashrate(),
            "mempool_size": len(chain.mempool),
            "total_transactions": chain.get_total_transactions(),
            "block_time_avg": chain.get_average_block_time(),
            "network_id": node.network.get_network_id()
        })
        
        return network_stats
    except Exception as e:
        logger.error(f"Error getting network stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility Endpoints
@router.get("/config")
async def get_config(
    request: Request,
    section: Optional[str] = None,
    auth: bool = Depends(authenticate_request)
):
    """Get node configuration"""
    node = request.app.state.node
    try:
        if section:
            config_value = node.config_manager.get(section)
            return {section: config_value}
        else:
            return node.config_manager.get_all_config()
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(request: Request):
    """Comprehensive health check endpoint"""
    node = request.app.state.node
    try:
        health_status = {
            "status": "healthy",
            "timestamp": int(datetime.now().timestamp()),
            "service": "rayonix-node",
            "version": "1.0.0"
        }
        
        # Check component health
        components = {
            "blockchain": node.rayonix_chain is not None,
            "config": node.config_manager is not None,
            "state_manager": node.state_manager is not None,
            "network": node.network is not None,
            "api": node.api_server is not None,
            "wallet": node.wallet is not None
        }
        
        health_status["components"] = components
        
        # Check if any critical component is unhealthy
        critical_components = ["blockchain", "config", "state_manager"]
        unhealthy_components = [comp for comp in critical_components if not components[comp]]
        
        if unhealthy_components:
            health_status["status"] = "unhealthy"
            health_status["unhealthy_components"] = unhealthy_components
        
        # Add performance metrics
        health_status["performance"] = {
            "uptime": node.state_manager.get_uptime(),
            "memory_usage": node.state_manager.get_memory_usage(),
            "cpu_usage": node.state_manager.get_cpu_usage(),
            "block_height": node.rayonix_chain.get_block_count() if node.rayonix_chain else 0,
            "peers_connected": len(await node.network.get_peers()) if node.network else 0
        }
        
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": int(datetime.now().timestamp())
        }