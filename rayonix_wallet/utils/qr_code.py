import qrcode
import json
import time
import zlib
import base64
from io import BytesIO
from typing import Optional, Dict, List, Tuple, Union
from enum import Enum
import hashlib
import hmac

class QRErrorCorrection(Enum):
    """QR code error correction levels"""
    LOW = "L"
    MEDIUM = "M"
    QUARTILE = "Q"
    HIGH = "H"

class QRCodeType(Enum):
    """Types of QR codes supported"""
    PAYMENT = "payment"
    WALLET_BACKUP = "wallet_backup"
    AUTH = "authentication"
    MULTIPART = "multipart"

class QRCodeVersion(Enum):
    """QR code versions for data structure"""
    V1 = "1.0"
    V2 = "2.0"

class QRCodeGenerator:
    """Production-ready QR code generator with advanced features"""
    
    def __init__(self, 
                 default_size: int = 15,
                 default_border: int = 2,
                 default_error_correction: QRErrorCorrection = QRErrorCorrection.HIGH):
        self.default_size = default_size
        self.default_border = default_border
        self.default_error_correction = default_error_correction
        
        # Error correction mapping
        self.error_correction_map = {
            QRErrorCorrection.LOW: qrcode.constants.ERROR_CORRECT_L,
            QRErrorCorrection.MEDIUM: qrcode.constants.ERROR_CORRECT_M,
            QRErrorCorrection.QUARTILE: qrcode.constants.ERROR_CORRECT_Q,
            QRErrorCorrection.HIGH: qrcode.constants.ERROR_CORRECT_H
        }
        
        # QR code configuration
        self.max_data_size = 2953  # Maximum bytes for QR code version 40 with H correction
        self.multipart_chunk_size = 1000  # Conservative chunk size for multipart

    def generate_qr_code(self, 
                        data: str, 
                        size: Optional[int] = None,
                        border: Optional[int] = None,
                        error_correction: Optional[QRErrorCorrection] = None,
                        fill_color: str = "black",
                        back_color: str = "white",
                        optimize: bool = True) -> bytes:
        """
        Generate high-quality QR code with advanced options
        
        Args:
            data: Data to encode
            size: Box size in pixels
            border: Border size in boxes
            error_correction: Error correction level
            fill_color: QR code color
            back_color: Background color
            optimize: Optimize data encoding
        
        Returns:
            PNG image bytes
        """
        try:
            # Use defaults if not provided
            size = size or self.default_size
            border = border or self.default_border
            error_correction = error_correction or self.default_error_correction
            
            ec_level = self.error_correction_map[error_correction]
            
            # Create QR code with optimized settings
            qr = qrcode.QRCode(
                version=None,  # Auto-detect version
                error_correction=ec_level,
                box_size=size,
                border=border,
                image_factory=None
            )
            
            # Add data with optimization
            if optimize:
                qr.add_data(data, optimize=0)
            else:
                qr.add_data(data)
            
            qr.make(fit=True)
            
            # Create image with high quality
            img = qr.make_image(
                fill_color=fill_color,
                back_color=back_color,
                image_factory=None
            )
            
            # Convert to high-quality PNG
            buffer = BytesIO()
            img.save(buffer, format="PNG", optimize=True, quality=95)
            return buffer.getvalue()
            
        except Exception as e:
            raise WalletError(f"QR code generation failed: {str(e)}")

    def generate_payment_qr_code(self, 
                               address: str, 
                               amount: Optional[float] = None,
                               currency: str = "RAY",
                               memo: Optional[str] = None, 
                               network: str = "mainnet",
                               recipient_name: Optional[str] = None,
                               payment_id: Optional[str] = None) -> bytes:
        """
        Generate advanced payment QR code with metadata
        
        Args:
            address: Recipient address
            amount: Payment amount
            currency: Currency code
            memo: Payment memo
            network: Network type
            recipient_name: Recipient name
            payment_id: Unique payment identifier
        
        Returns:
            PNG image bytes
        """
        try:
            # Create comprehensive payment data structure
            payment_data = {
                "version": QRCodeVersion.V2.value,
                "type": QRCodeType.PAYMENT.value,
                "network": network,
                "currency": currency.upper(),
                "address": address,
                "timestamp": int(time.time())
            }
            
            # Add optional fields
            if amount is not None:
                payment_data["amount"] = str(amount)  # String to avoid float precision issues
            
            if memo:
                payment_data["memo"] = memo[:140]  # Limit memo length
            
            if recipient_name:
                payment_data["recipient"] = recipient_name[:50]
            
            if payment_id:
                payment_data["payment_id"] = payment_id
            
            # Add data integrity check
            payment_data["checksum"] = self._generate_checksum(payment_data)
            
            # Convert to JSON
            json_data = json.dumps(payment_data, separators=(',', ':'))
            
            return self.generate_qr_code(json_data)
            
        except Exception as e:
            raise WalletError(f"Payment QR code generation failed: {str(e)}")

    def generate_wallet_backup_qr(self, 
                                mnemonic: str, 
                                passphrase: Optional[str] = None,
                                wallet_name: Optional[str] = None,
                                derivation_path: Optional[str] = None,
                                compress: bool = True) -> Union[bytes, List[bytes]]:
        """
        Generate secure wallet backup QR code(s)
        
        Args:
            mnemonic: Wallet mnemonic phrase
            passphrase: Optional passphrase
            wallet_name: Wallet identifier
            derivation_path: Derivation path
            compress: Enable data compression
        
        Returns:
            Single QR code bytes or list of multipart QR codes
        """
        try:
            # Create secure backup data structure
            backup_data = {
                "version": QRCodeVersion.V2.value,
                "type": QRCodeType.WALLET_BACKUP.value,
                "timestamp": int(time.time()),
                "mnemonic": mnemonic,
                "wallet_id": self._generate_wallet_id(mnemonic)
            }
            
            # Add optional metadata
            if passphrase:
                backup_data["passphrase"] = self._encrypt_sensitive(passphrase, mnemonic)
            
            if wallet_name:
                backup_data["name"] = wallet_name
            
            if derivation_path:
                backup_data["derivation_path"] = derivation_path
            
            # Add security metadata
            backup_data["checksum"] = self._generate_checksum(backup_data)
            backup_data["data_hash"] = self._generate_data_hash(backup_data)
            
            # Convert to JSON
            json_data = json.dumps(backup_data, separators=(',', ':'))
            
            # Compress if requested
            if compress:
                json_data = self._compress_data(json_data)
            
            # Check if multipart is needed
            if len(json_data.encode('utf-8')) > self.max_data_size:
                return self._generate_multipart_qr_codes(json_data, QRCodeType.WALLET_BACKUP)
            
            return self.generate_qr_code(json_data)
            
        except Exception as e:
            raise WalletError(f"Wallet backup QR generation failed: {str(e)}")

    def read_qr_code(self, 
                    image_data: bytes, 
                    validate_checksum: bool = True) -> Optional[Dict]:
        """
        Read and validate QR code data with advanced processing
        
        Args:
            image_data: QR code image bytes
            validate_checksum: Enable checksum validation
        
        Returns:
            Parsed and validated data dictionary
        """
        try:
            import cv2
            import numpy as np
            
            # Convert bytes to numpy array with error handling
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise WalletError("Invalid image data")
            
            # Initialize advanced QR code detector
            detector = cv2.QRCodeDetector()
            
            # Detect with enhanced parameters
            data, vertices, straight_qrcode = detector.detectAndDecode(img)
            
            if not data:
                # Try with preprocessing for difficult QR codes
                enhanced_img = self._enhance_image(img)
                data, vertices, straight_qrcode = detector.detectAndDecode(enhanced_img)
            
            if data:
                # Parse and validate data
                parsed_data = self._parse_qr_data(data, validate_checksum)
                return parsed_data
            
            return None
            
        except Exception as e:
            raise WalletError(f"QR code reading failed: {str(e)}")

    def validate_qr_code_data(self, 
                            data: Dict, 
                            expected_type: Optional[QRCodeType] = None,
                            max_age: Optional[int] = None) -> Tuple[bool, List[str]]:
        """
        Comprehensive QR code data validation
        
        Args:
            data: Parsed QR code data
            expected_type: Expected QR code type
            max_age: Maximum age in seconds
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check basic structure
            if not isinstance(data, dict):
                errors.append("Data must be a dictionary")
                return False, errors
            
            # Validate version
            if "version" not in data:
                errors.append("Missing version field")
            elif data["version"] not in [v.value for v in QRCodeVersion]:
                errors.append(f"Unsupported version: {data['version']}")
            
            # Validate type
            if "type" not in data:
                errors.append("Missing type field")
            elif expected_type and data["type"] != expected_type.value:
                errors.append(f"Expected type {expected_type.value}, got {data['type']}")
            elif data["type"] not in [t.value for t in QRCodeType]:
                errors.append(f"Invalid type: {data['type']}")
            
            # Validate timestamp
            if "timestamp" in data and max_age:
                current_time = int(time.time())
                if current_time - data["timestamp"] > max_age:
                    errors.append("QR code data is too old")
            
            # Type-specific validation
            if data.get("type") == QRCodeType.WALLET_BACKUP.value:
                if "mnemonic" not in data:
                    errors.append("Missing mnemonic in wallet backup")
                if "checksum" not in data:
                    errors.append("Missing checksum in wallet backup")
            
            elif data.get("type") == QRCodeType.PAYMENT.value:
                if "address" not in data:
                    errors.append("Missing address in payment data")
                if "network" not in data:
                    errors.append("Missing network in payment data")
            
            # Validate checksum if present
            if "checksum" in data and not self._validate_checksum(data):
                errors.append("Checksum validation failed")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors

    def _generate_multipart_qr_codes(self, data: str, data_type: QRCodeType) -> List[bytes]:
        """Generate multiple QR codes for large data with advanced handling"""
        try:
            # Split data into optimized chunks
            chunks = self._split_data_optimally(data, self.multipart_chunk_size)
            total_parts = len(chunks)
            
            qr_codes = []
            for part_num, chunk in enumerate(chunks, 1):
                multipart_data = {
                    "version": QRCodeVersion.V2.value,
                    "type": QRCodeType.MULTIPART.value,
                    "data_type": data_type.value,
                    "part": part_num,
                    "total_parts": total_parts,
                    "data": chunk,
                    "timestamp": int(time.time()),
                    "data_hash": self._generate_data_hash(data)
                }
                
                multipart_data["checksum"] = self._generate_checksum(multipart_data)
                json_data = json.dumps(multipart_data, separators=(',', ':'))
                
                qr_code = self.generate_qr_code(json_data)
                qr_codes.append(qr_code)
            
            return qr_codes
            
        except Exception as e:
            raise WalletError(f"Multipart QR generation failed: {str(e)}")

    def _split_data_optimally(self, data: str, chunk_size: int) -> List[str]:
        """Split data into chunks considering JSON structure"""
        chunks = []
        current_chunk = ""
        
        # Simple splitting for now - can be enhanced with JSON-aware splitting
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks

    def _generate_checksum(self, data: Dict) -> str:
        """Generate SHA-256 checksum for data integrity"""
        data_copy = data.copy()
        data_copy.pop('checksum', None)  # Remove existing checksum
        
        json_str = json.dumps(data_copy, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]  # Truncate for QR space

    def _validate_checksum(self, data: Dict) -> bool:
        """Validate data checksum"""
        if 'checksum' not in data:
            return False
        
        stored_checksum = data['checksum']
        calculated_checksum = self._generate_checksum(data)
        
        return stored_checksum == calculated_checksum

    def _generate_data_hash(self, data: Union[Dict, str]) -> str:
        """Generate hash of data content"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()[:8]

    def _generate_wallet_id(self, mnemonic: str) -> str:
        """Generate deterministic wallet ID from mnemonic"""
        return hashlib.sha256(mnemonic.encode('utf-8')).hexdigest()[:12]

    def _encrypt_sensitive(self, data: str, key: str) -> str:
        """Simple XOR encryption for sensitive data (basic protection)"""
        # Note: For production, use proper encryption
        encrypted = ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(data, key * len(data)))
        return base64.b64encode(encrypted.encode('utf-8')).decode('utf-8')

    def _compress_data(self, data: str) -> str:
        """Compress data using zlib"""
        compressed = zlib.compress(data.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')

    def _decompress_data(self, compressed_data: str) -> str:
        """Decompress zlib-compressed data"""
        decompressed = zlib.decompress(base64.b64decode(compressed_data))
        return decompressed.decode('utf-8')

    def _enhance_image(self, img) -> any:
        """Enhance image for better QR code detection"""
        import cv2
        import numpy as np
        
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(img)
        
        # Apply Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply thresholding
        _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return enhanced

    def _parse_qr_data(self, data: str, validate_checksum: bool) -> Dict:
        """Parse and process QR code data"""
        try:
            # Try to parse as JSON
            parsed_data = json.loads(data)
            
            # Decompress if necessary
            if isinstance(parsed_data, dict) and parsed_data.get('compressed'):
                parsed_data = self._decompress_data(parsed_data['data'])
                parsed_data = json.loads(parsed_data)
            
            # Validate checksum
            if validate_checksum and not self._validate_checksum(parsed_data):
                raise WalletError("QR code data checksum validation failed")
            
            return parsed_data
            
        except json.JSONDecodeError:
            # Handle plain text data (backward compatibility)
            return {"data": data, "type": "legacy", "version": "1.0"}

# Backward compatibility functions
def generate_qr_code(data: str, size: int = 10, border: int = 4, error_correction: str = "H") -> bytes:
    generator = QRCodeGenerator()
    ec_level = QRErrorCorrection(error_correction.upper())
    return generator.generate_qr_code(data, size, border, ec_level)

def generate_payment_qr_code(address: str, amount: Optional[float] = None, 
                           memo: Optional[str] = None, network: str = "mainnet") -> bytes:
    generator = QRCodeGenerator()
    return generator.generate_payment_qr_code(address, amount, memo=memo, network=network)

def read_qr_code(image_data: bytes) -> Optional[str]:
    generator = QRCodeGenerator()
    result = generator.read_qr_code(image_data, validate_checksum=False)
    return json.dumps(result) if result else None

def generate_wallet_backup_qr(mnemonic: str, passphrase: Optional[str] = None) -> bytes:
    generator = QRCodeGenerator()
    result = generator.generate_wallet_backup_qr(mnemonic, passphrase)
    return result[0] if isinstance(result, list) else result

def validate_qr_code_data(data: str, expected_type: Optional[str] = None) -> bool:
    generator = QRCodeGenerator()
    try:
        parsed_data = json.loads(data)
        qr_type = QRCodeType(expected_type) if expected_type else None
        is_valid, errors = generator.validate_qr_code_data(parsed_data, qr_type)
        return is_valid
    except:
        return False

# Custom exception (assuming this exists in the original context)
class WalletError(Exception):
    pass