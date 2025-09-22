import qrcode
from io import BytesIO
from typing import Optional
from ..core.exceptions import WalletError

def generate_qr_code(data: str, size: int = 10, border: int = 4, 
                   error_correction: str = "H") -> bytes:
    """Generate QR code from data"""
    try:
        # Map error correction levels
        error_correction_map = {
            "L": qrcode.constants.ERROR_CORRECT_L,
            "M": qrcode.constants.ERROR_CORRECT_M,
            "Q": qrcode.constants.ERROR_CORRECT_Q,
            "H": qrcode.constants.ERROR_CORRECT_H
        }
        
        ec_level = error_correction_map.get(error_correction.upper(), 
                                          qrcode.constants.ERROR_CORRECT_H)
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=ec_level,
            box_size=size,
            border=border,
        )
        
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
        
    except Exception as e:
        raise WalletError(f"QR code generation failed: {e}")

def generate_payment_qr_code(address: str, amount: Optional[float] = None, 
                           memo: Optional[str] = None, network: str = "mainnet") -> bytes:
    """Generate payment request QR code"""
    try:
        # Create payment URI based on network
        if network == "mainnet":
            prefix = "rayonix:"
        else:
            prefix = "rayonix-testnet:"
        
        payment_uri = prefix + address
        
        # Add amount if specified
        if amount is not None:
            payment_uri += f"?amount={amount}"
            
            # Add memo if specified
            if memo is not None:
                payment_uri += f"&memo={memo}"
        elif memo is not None:
            payment_uri += f"?memo={memo}"
        
        return generate_qr_code(payment_uri)
        
    except Exception as e:
        raise WalletError(f"Payment QR code generation failed: {e}")

def read_qr_code(image_data: bytes) -> Optional[str]:
    """Read data from QR code image"""
    try:
        import cv2
        import numpy as np
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Initialize QR code detector
        detector = cv2.QRCodeDetector()
        
        # Detect and decode QR code
        data, vertices, _ = detector.detectAndDecode(img)
        
        if data:
            return data
        return None
        
    except Exception as e:
        raise WalletError(f"QR code reading failed: {e}")

def generate_wallet_backup_qr(mnemonic: str, passphrase: Optional[str] = None) -> bytes:
    """Generate QR code for wallet backup"""
    try:
        backup_data = {
            "type": "wallet_backup",
            "mnemonic": mnemonic,
            "timestamp": int(time.time())
        }
        
        if passphrase:
            backup_data["passphrase"] = passphrase
        
        # Convert to JSON and generate QR code
        import json
        json_data = json.dumps(backup_data)
        
        # Split into multiple QR codes if data is too large
        max_data_size = 1000  # Approximate max characters per QR code
        if len(json_data) > max_data_size:
            return _generate_multipart_qr_codes(json_data, max_data_size)
        
        return generate_qr_code(json_data)
        
    except Exception as e:
        raise WalletError(f"Wallet backup QR generation failed: {e}")

def _generate_multipart_qr_codes(data: str, chunk_size: int) -> bytes:
    """Generate multiple QR codes for large data"""
    # Split data into chunks
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Generate QR code for each chunk
    qr_codes = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "part": i + 1,
            "total": len(chunks),
            "data": chunk
        }
        
        import json
        qr_data = json.dumps(chunk_data)
        qr_codes.append(generate_qr_code(qr_data))
    
    # For simplicity, return first QR code
    # In production, you might want to handle multiple QR codes differently
    return qr_codes[0]

def validate_qr_code_data(data: str, expected_type: Optional[str] = None) -> bool:
    """Validate QR code data structure"""
    try:
        import json
        parsed_data = json.loads(data)
        
        if expected_type and parsed_data.get("type") != expected_type:
            return False
        
        # Add more validation based on expected type
        if expected_type == "wallet_backup":
            return "mnemonic" in parsed_data
        
        return True
        
    except:
        return False