# smart_contract/wasm/bytecode_validator.py
import logging
import hashlib
import struct
import zlib
from typing import Dict, Any, Optional, List, Set, Tuple
from enum import Enum
from dataclasses import dataclass
import re

logger = logging.getLogger("SmartContract.WASMValidator")

class ValidationLevel(Enum):
    """Validation level enumeration"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"

class SecurityThreatLevel(Enum):
    """Security threat level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Detailed validation result"""
    is_valid: bool
    threats_detected: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, Any]
    validation_time: float
    bytecode_hash: str

@dataclass
class ValidationConfig:
    """Comprehensive validation configuration"""
    max_module_size: int = 10 * 1024 * 1024  # 10MB
    max_memory_pages: int = 65536  # 4GB
    max_table_size: int = 100000
    max_functions: int = 10000
    max_globals: int = 1000
    max_data_segments: int = 10000
    max_element_segments: int = 10000
    max_imports: int = 1000
    max_exports: int = 1000
    validation_level: ValidationLevel = ValidationLevel.STRICT
    enable_heuristic_analysis: bool = True
    enable_pattern_matching: bool = True
    enable_entropy_analysis: bool = True
    enable_control_flow_analysis: bool = True
    enable_resource_estimation: bool = True

class WASMBytecodeValidator:
    """Production-grade WebAssembly bytecode validator with comprehensive security checks"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        # Enhanced malicious patterns database
        self.malicious_patterns = self._initialize_malicious_patterns()
        
        # WASM specification constants
        self.WASM_MAGIC = b'\x00\x61\x73\x6D'  # \0asm
        self.WASM_VERSION = b'\x01\x00\x00\x00'  # Version 1
        
        # Comprehensive opcode database
        self.opcode_categories = self._initialize_opcode_categories()
        self.restricted_opcodes = self._initialize_restricted_opcodes()
        
        # Section identifiers
        self.SECTION_IDS = {
            0: 'custom',
            1: 'type',
            2: 'import',
            3: 'function',
            4: 'table',
            5: 'memory',
            6: 'global',
            7: 'export',
            8: 'start',
            9: 'element',
            10: 'code',
            11: 'data'
        }
        
        # Validation cache
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Threat intelligence
        self.known_vulnerabilities = self._initialize_known_vulnerabilities()
        
        # Performance metrics
        self.metrics = {
            'validations_performed': 0,
            'threats_detected': 0,
            'modules_rejected': 0,
            'modules_accepted': 0,
            'average_validation_time': 0.0,
            'total_validation_time': 0.0
        }
        
        logger.info("Advanced WASMBytecodeValidator initialized")

    def _initialize_malicious_patterns(self) -> Dict[bytes, SecurityThreatLevel]:
        """Initialize comprehensive malicious patterns database"""
        return {
            # Code injection patterns
            b'eval': SecurityThreatLevel.HIGH,
            b'exec': SecurityThreatLevel.HIGH,
            b'system': SecurityThreatLevel.HIGH,
            b'__asm__': SecurityThreatLevel.MEDIUM,
            
            # Memory exploitation patterns
            b'buffer_overflow': SecurityThreatLevel.CRITICAL,
            b'heap_spray': SecurityThreatLevel.CRITICAL,
            b'use_after_free': SecurityThreatLevel.CRITICAL,
            
            # Resource exhaustion patterns
            b'infinite_loop': SecurityThreatLevel.HIGH,
            b'memory_exhaustion': SecurityThreatLevel.HIGH,
            b'stack_overflow': SecurityThreatLevel.HIGH,
            
            # Cryptographic mining patterns
            b'cryptonight': SecurityThreatLevel.MEDIUM,
            b'scrypt': SecurityThreatLevel.MEDIUM,
            b'sha256': SecurityThreatLevel.LOW,
            
            # Obfuscation patterns
            b'packed': SecurityThreatLevel.MEDIUM,
            b'obfuscated': SecurityThreatLevel.MEDIUM,
            b'encrypted': SecurityThreatLevel.MEDIUM,
            
            # Network patterns
            b'socket': SecurityThreatLevel.HIGH,
            b'connect': SecurityThreatLevel.HIGH,
            b'http': SecurityThreatLevel.MEDIUM,
        }

    def _initialize_opcode_categories(self) -> Dict[int, str]:
        """Initialize comprehensive opcode categorization"""
        return {
            # Control flow opcodes
            0x00: 'unreachable', 0x01: 'nop', 0x02: 'block', 0x03: 'loop',
            0x04: 'if', 0x05: 'else', 0x0B: 'end', 0x0C: 'br', 0x0D: 'br_if',
            0x0E: 'br_table', 0x0F: 'return', 0x10: 'call', 0x11: 'call_indirect',
            
            # Parametric opcodes
            0x1A: 'drop', 0x1B: 'select',
            
            # Variable access opcodes
            0x20: 'local.get', 0x21: 'local.set', 0x22: 'local.tee',
            0x23: 'global.get', 0x24: 'global.set',
            
            # Memory-related opcodes
            0x28: 'i32.load', 0x29: 'i64.load', 0x2A: 'f32.load', 0x2B: 'f64.load',
            0x2C: 'i32.load8_s', 0x2D: 'i32.load8_u', 0x2E: 'i32.load16_s', 0x2F: 'i32.load16_u',
            0x36: 'i32.store', 0x37: 'i64.store', 0x38: 'f32.store', 0x39: 'f64.store',
            0x3A: 'i32.store8', 0x3B: 'i32.store16', 0x3F: 'memory.size', 0x40: 'memory.grow',
            
            # Numeric opcodes
            0x41: 'i32.const', 0x42: 'i64.const', 0x43: 'f32.const', 0x44: 'f64.const',
            
            # Integer arithmetic
            0x45: 'i32.eqz', 0x46: 'i32.eq', 0x47: 'i32.ne', 0x48: 'i32.lt_s',
            # ... more numeric opcodes
        }

    def _initialize_restricted_opcodes(self) -> Set[int]:
        """Initialize restricted opcodes based on validation level"""
        base_restricted = {
            0x00,  # unreachable - can cause undefined behavior
            0x08,  # deprecated opcodes
            0x09,  # deprecated opcodes
        }
        
        if self.config.validation_level == ValidationLevel.PARANOID:
            base_restricted.update({
                0x40,  # memory.grow - potential memory exhaustion
                0x3F,  # memory.size - information disclosure
                0x11,  # call_indirect - potential code injection
            })
            
        return base_restricted

    def _initialize_known_vulnerabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of known WASM vulnerabilities"""
        return {
            'CVE-2021-XXX': {
                'description': 'WASM memory corruption vulnerability',
                'pattern': b'\x41\x00\x28\x02\x00\x1A',
                'threat_level': SecurityThreatLevel.CRITICAL,
                'affected_versions': ['1.0'],
            },
            'CVE-2022-XXX': {
                'description': 'Type confusion in function calls',
                'pattern': b'\x10\x00\x20\x00\x6A',
                'threat_level': SecurityThreatLevel.HIGH,
                'affected_versions': ['1.0'],
            }
        }

    def validate_bytecode(self, bytecode: bytes, 
                         context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Comprehensive WASM bytecode validation with detailed reporting"""
        start_time = time.time()
        
        # Check cache first
        bytecode_hash = self._calculate_bytecode_hash(bytecode)
        cache_key = f"{bytecode_hash}_{self.config.validation_level.value}"
        
        if cache_key in self.validation_cache:
            self.cache_hits += 1
            return self.validation_cache[cache_key]
        
        self.cache_misses += 1
        context = context or {}
        
        result = ValidationResult(
            is_valid=False,
            threats_detected=[],
            warnings=[],
            errors=[],
            metrics={},
            validation_time=0.0,
            bytecode_hash=bytecode_hash
        )
        
        try:
            # Multi-stage validation pipeline
            validation_stages = [
                self._validate_basic_structure,
                self._validate_wasm_header,
                self._validate_sections,
                self._validate_imports_exports,
                self._validate_memory_sections,
                self._validate_code_sections,
                self._validate_resource_limits,
                self._perform_security_analysis,
                self._perform_heuristic_analysis,
            ]
            
            if self.config.enable_control_flow_analysis:
                validation_stages.append(self._analyze_control_flow)
                
            if self.config.enable_entropy_analysis:
                validation_stages.append(self._analyze_entropy)
            
            # Execute validation pipeline
            for stage in validation_stages:
                stage_result = stage(bytecode, context)
                if not stage_result['valid']:
                    result.errors.extend(stage_result.get('errors', []))
                    result.warnings.extend(stage_result.get('warnings', []))
                    result.threats_detected.extend(stage_result.get('threats', []))
                    
                    if stage_result.get('fatal', False):
                        break
            
            # Determine overall validity
            result.is_valid = (
                len(result.errors) == 0 and 
                len([t for t in result.threats_detected 
                    if t['threat_level'] in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]]) == 0
            )
            
            # Update metrics
            result.metrics = self._calculate_module_metrics(bytecode)
            result.validation_time = time.time() - start_time
            
            # Cache the result
            self.validation_cache[cache_key] = result
            self._update_global_metrics(result)
            
            logger.info(f"Bytecode validation completed: valid={result.is_valid}, "
                       f"threats={len(result.threats_detected)}, "
                       f"time={result.validation_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Validation process failed: {e}")
            result.errors.append(f"Validation process error: {e}")
            result.validation_time = time.time() - start_time
        
        return result

    def _validate_basic_structure(self, bytecode: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic bytecode structure"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        if not bytecode:
            result['valid'] = False
            result['errors'].append("Empty bytecode provided")
            return result
        
        if len(bytecode) < 8:
            result['valid'] = False
            result['errors'].append("Bytecode too short to be valid WASM module")
            return result
        
        if len(bytecode) > self.config.max_module_size:
            result['valid'] = False
            result['errors'].append(
                f"Module size {len(bytecode)} exceeds maximum allowed {self.config.max_module_size}"
            )
            return result
        
        # Check for obvious corruption
        if self._detect_corruption(bytecode):
            result['valid'] = False
            result['errors'].append("Bytecode appears to be corrupted")
            return result
        
        return result

    def _validate_wasm_header(self, bytecode: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate WASM magic number and version"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Check magic number
            if len(bytecode) < 4 or bytecode[:4] != self.WASM_MAGIC:
                result['valid'] = False
                result['errors'].append("Invalid WASM magic number")
                return result
            
            # Check version
            if len(bytecode) < 8 or bytecode[4:8] != self.WASM_VERSION:
                result['valid'] = False
                result['errors'].append("Unsupported WASM version")
                return result
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Header validation failed: {e}")
            
        return result

    def _validate_sections(self, bytecode: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate WASM module sections"""
        result = {'valid': True, 'errors': [], 'warnings': [], 'threats': []}
        
        try:
            pos = 8  # Start after header
            section_counts = {}
            
            while pos < len(bytecode):
                if pos + 1 >= len(bytecode):
                    result['warnings'].append("Unexpected end of bytecode in section header")
                    break
                
                section_id = bytecode[pos]
                pos += 1
                
                # Read section size (LEB128)
                section_size, bytes_read = self._read_leb128(bytecode, pos)
                if bytes_read == 0:
                    result['errors'].append("Failed to read section size")
                    result['valid'] = False
                    break
                    
                pos += bytes_read
                
                # Validate section
                section_name = self.SECTION_IDS.get(section_id, f'unknown_{section_id}')
                section_counts[section_name] = section_counts.get(section_name, 0) + 1
                
                # Check for duplicate sections (except custom sections)
                if section_id != 0 and section_counts[section_name] > 1:
                    result['warnings'].append(f"Duplicate {section_name} section detected")
                
                # Validate section content
                section_end = pos + section_size
                if section_end > len(bytecode):
                    result['errors'].append(f"{section_name} section extends beyond bytecode")
                    result['valid'] = False
                    break
                
                # Section-specific validation
                section_validation = self._validate_specific_section(
                    section_id, bytecode[pos:section_end], context
                )
                if not section_validation['valid']:
                    result['errors'].extend(section_validation['errors'])
                    result['threats'].extend(section_validation.get('threats', []))
                    result['valid'] = section_validation.get('fatal', True) and result['valid']
                
                pos = section_end
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Section validation failed: {e}")
            
        return result

    def _validate_specific_section(self, section_id: int, section_data: bytes, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate specific section types"""
        result = {'valid': True, 'errors': [], 'threats': []}
        
        try:
            if section_id == 2:  # Import section
                result.update(self._validate_imports_section(section_data))
            elif section_id == 3:  # Function section
                result.update(self._validate_functions_section(section_data))
            elif section_id == 5:  # Memory section
                result.update(self._validate_memory_section(section_data))
            elif section_id == 7:  # Export section
                result.update(self._validate_exports_section(section_data))
            elif section_id == 10:  # Code section
                result.update(self._validate_code_section(section_data))
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Section {section_id} validation error: {e}")
            
        return result

    def _validate_imports_section(self, section_data: bytes) -> Dict[str, Any]:
        """Validate imports section for security concerns"""
        result = {'valid': True, 'warnings': [], 'threats': []}
        
        try:
            import_count, pos = self._read_leb128(section_data, 0)
            
            if import_count > self.config.max_imports:
                result['valid'] = False
                result['threats'].append({
                    'type': 'EXCESSIVE_IMPORTS',
                    'threat_level': SecurityThreatLevel.MEDIUM,
                    'description': f'Too many imports: {import_count}'
                })
                
            # Analyze individual imports
            for i in range(import_count):
                if pos >= len(section_data):
                    result['warnings'].append("Unexpected end in imports section")
                    break
                    
                # Read module name
                module_len, bytes_read = self._read_leb128(section_data, pos)
                pos += bytes_read
                module_name = section_data[pos:pos+module_len].decode('utf-8', errors='ignore')
                pos += module_len
                
                # Check for suspicious import modules
                if module_name not in ['env', 'wasi_snapshot_preview1']:
                    result['threats'].append({
                        'type': 'SUSPICIOUS_IMPORT_MODULE',
                        'threat_level': SecurityThreatLevel.MEDIUM,
                        'description': f'Suspicious import module: {module_name}'
                    })
                    
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Imports section validation failed: {e}")
            
        return result

    def _validate_memory_section(self, section_data: bytes) -> Dict[str, Any]:
        """Validate memory section for resource limits"""
        result = {'valid': True, 'errors': [], 'threats': []}
        
        try:
            memory_count, pos = self._read_leb128(section_data, 0)
            
            if memory_count > 1:
                result['threats'].append({
                    'type': 'MULTIPLE_MEMORIES',
                    'threat_level': SecurityThreatLevel.LOW,
                    'description': 'Multiple memory sections defined'
                })
                
            for i in range(memory_count):
                if pos >= len(section_data):
                    break
                    
                # Read memory limits
                flags = section_data[pos]
                pos += 1
                
                initial_pages, bytes_read = self._read_leb128(section_data, pos)
                pos += bytes_read
                
                if flags & 0x1:  # Has maximum
                    maximum_pages, bytes_read = self._read_leb128(section_data, pos)
                    pos += bytes_read
                    
                    if maximum_pages > self.config.max_memory_pages:
                        result['valid'] = False
                        result['errors'].append(
                            f"Memory limit {maximum_pages} pages exceeds maximum {self.config.max_memory_pages}"
                        )
                        
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Memory section validation failed: {e}")
            
        return result

    def _validate_code_section(self, section_data: bytes) -> Dict[str, Any]:
        """Validate code section for security threats"""
        result = {'valid': True, 'errors': [], 'threats': []}
        
        try:
            function_count, pos = self._read_leb128(section_data, 0)
            
            if function_count > self.config.max_functions:
                result['valid'] = False
                result['errors'].append(
                    f"Too many functions: {function_count} > {self.config.max_functions}"
                )
                return result
                
            for i in range(function_count):
                if pos >= len(section_data):
                    result['warnings'].append("Unexpected end in code section")
                    break
                    
                # Read function body size
                body_size, bytes_read = self._read_leb128(section_data, pos)
                pos += bytes_read
                
                if pos + body_size > len(section_data):
                    result['errors'].append("Function body extends beyond section")
                    result['valid'] = False
                    break
                    
                # Validate function body
                function_body = section_data[pos:pos+body_size]
                function_validation = self._validate_function_body(function_body)
                
                if not function_validation['valid']:
                    result['threats'].extend(function_validation.get('threats', []))
                    result['valid'] = result['valid'] and function_validation['valid']
                
                pos += body_size
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Code section validation failed: {e}")
            
        return result

    def _validate_function_body(self, body: bytes) -> Dict[str, Any]:
        """Validate individual function body"""
        result = {'valid': True, 'threats': []}
        
        try:
            # Check for restricted opcodes
            for opcode in self.restricted_opcodes:
                if opcode in body:
                    result['threats'].append({
                        'type': 'RESTRICTED_OPCODE',
                        'threat_level': SecurityThreatLevel.MEDIUM,
                        'description': f'Restricted opcode 0x{opcode:02X} detected'
                    })
            
            # Check for suspicious opcode sequences
            suspicious_sequences = [
                (0x41, 0x10),  # const followed by call - potential code injection
                (0x40, 0x40),  # consecutive memory.grow - memory exhaustion
            ]
            
            for seq in suspicious_sequences:
                if self._find_byte_sequence(body, seq):
                    result['threats'].append({
                        'type': 'SUSPICIOUS_OPCODE_SEQUENCE',
                        'threat_level': SecurityThreatLevel.LOW,
                        'description': f'Suspicious opcode sequence detected'
                    })
                    
        except Exception as e:
            result['valid'] = False
            
        return result

    def _perform_security_analysis(self, bytecode: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive security analysis"""
        result = {'valid': True, 'threats': []}
        
        # Pattern matching against known threats
        if self.config.enable_pattern_matching:
            for pattern, threat_level in self.malicious_patterns.items():
                if pattern in bytecode:
                    result['threats'].append({
                        'type': 'MALICIOUS_PATTERN',
                        'threat_level': threat_level,
                        'description': f'Known malicious pattern detected: {pattern}',
                        'pattern': pattern.hex()
                    })
        
        # Check for known vulnerabilities
        for vuln_id, vuln_info in self.known_vulnerabilities.items():
            if vuln_info['pattern'] in bytecode:
                result['threats'].append({
                    'type': 'KNOWN_VULNERABILITY',
                    'threat_level': vuln_info['threat_level'],
                    'description': f'{vuln_id}: {vuln_info["description"]}',
                    'vulnerability_id': vuln_id
                })
        
        # Entropy analysis for packed/obfuscated code
        if self.config.enable_entropy_analysis:
            entropy = self._calculate_entropy(bytecode)
            if entropy > 7.5:  # High entropy indicates possible encryption
                result['threats'].append({
                    'type': 'HIGH_ENTROPY',
                    'threat_level': SecurityThreatLevel.MEDIUM,
                    'description': f'High entropy detected ({entropy:.2f}), possible obfuscation'
                })
        
        return result

    def _perform_heuristic_analysis(self, bytecode: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform heuristic analysis for advanced threats"""
        result = {'valid': True, 'warnings': [], 'threats': []}
        
        try:
            # Check for abnormal section ratios
            total_size = len(bytecode)
            if total_size > 0:
                code_section_ratio = bytecode.count(b'\x0A') / total_size  # Code section marker
                
                if code_section_ratio < 0.1:
                    result['warnings'].append("Abnormally small code section")
                elif code_section_ratio > 0.9:
                    result['warnings'].append("Abnormally large code section")
            
            # Check for compression artifacts
            if self._detects_compression(bytecode):
                result['threats'].append({
                    'type': 'COMPRESSION_DETECTED',
                    'threat_level': SecurityThreatLevel.LOW,
                    'description': 'Bytecode appears to be compressed'
                })
                
        except Exception as e:
            result['warnings'].append(f"Heuristic analysis incomplete: {e}")
            
        return result

    def _analyze_control_flow(self, bytecode: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze control flow for potential issues"""
        result = {'valid': True, 'threats': []}
        
        try:
            # Simple control flow analysis
            branch_instructions = bytecode.count(b'\x0C') + bytecode.count(b'\x0D') + bytecode.count(b'\x0E')
            total_instructions = max(1, len([b for b in bytecode if b in self.opcode_categories]))
            
            branch_ratio = branch_instructions / total_instructions
            
            if branch_ratio > 0.3:  # High branch density
                result['threats'].append({
                    'type': 'COMPLEX_CONTROL_FLOW',
                    'threat_level': SecurityThreatLevel.LOW,
                    'description': f'Complex control flow detected (branch ratio: {branch_ratio:.2f})'
                })
                
        except Exception as e:
            result['warnings'] = [f"Control flow analysis failed: {e}"]
            
        return result

    def _analyze_entropy(self, bytecode: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bytecode entropy for obfuscation detection"""
        result = {'valid': True, 'threats': []}
        
        try:
            entropy = self._calculate_entropy(bytecode)
            
            if entropy > 7.8:
                result['threats'].append({
                    'type': 'POSSIBLE_OBFUSCATION',
                    'threat_level': SecurityThreatLevel.MEDIUM,
                    'description': f'High entropy ({entropy:.2f}) suggests obfuscation'
                })
            elif entropy < 4.0:
                result['warnings'].append(f'Low entropy ({entropy:.2f}) detected')
                
        except Exception as e:
            result['warnings'] = [f"Entropy analysis failed: {e}"]
            
        return result

    # Utility methods
    def _read_leb128(self, data: bytes, pos: int) -> Tuple[int, int]:
        """Read LEB128 encoded integer"""
        result = 0
        shift = 0
        bytes_read = 0
        
        while pos < len(data):
            byte = data[pos]
            pos += 1
            bytes_read += 1
            
            result |= (byte & 0x7F) << shift
            shift += 7
            
            if not (byte & 0x80):
                break
                
        return result, bytes_read

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
            
        entropy = 0.0
        for x in range(256):
            p_x = float(data.count(x)) / len(data)
            if p_x > 0:
                entropy += - p_x * (p_x.bit_length() - 1)
                
        return entropy

    def _find_byte_sequence(self, data: bytes, sequence: Tuple[int, ...]) -> bool:
        """Find sequence of bytes in data"""
        if len(sequence) == 0:
            return False
            
        for i in range(len(data) - len(sequence) + 1):
            if all(data[i + j] == sequence[j] for j in range(len(sequence))):
                return True
        return False

    def _detect_corruption(self, bytecode: bytes) -> bool:
        """Detect obvious bytecode corruption"""
        # Check for all-zero sections
        if bytecode.count(b'\x00') / len(bytecode) > 0.9:
            return True
            
        # Check for repeating patterns indicating corruption
        if len(bytecode) > 100:
            sample = bytecode[:100]
            if sample.count(sample[0]) > 90:  # 90% same byte
                return True
                
        return False

    def _detects_compression(self, bytecode: bytes) -> bool:
        """Detect signs of compression"""
        # Check for common compression headers
        compression_signatures = [
            b'\x1F\x8B',      # GZIP
            b'\x78\x01',      # ZLIB
            b'\x78\x9C',      # ZLIB
            b'\xFD7zXZ',      # XZ
        ]
        
        return any(sig in bytecode for sig in compression_signatures)

    def _calculate_bytecode_hash(self, bytecode: bytes) -> str:
        """Calculate comprehensive bytecode hash"""
        return hashlib.sha256(bytecode).hexdigest()

    def _calculate_module_metrics(self, bytecode: bytes) -> Dict[str, Any]:
        """Calculate comprehensive module metrics"""
        return {
            'size': len(bytecode),
            'entropy': self._calculate_entropy(bytecode),
            'estimated_functions': bytecode.count(b'\x03'),  # Function section marker
            'estimated_memory_pages': bytecode.count(b'\x05'),  # Memory section marker
            'compression_detected': self._detects_compression(bytecode),
        }

    def _update_global_metrics(self, result: ValidationResult) -> None:
        """Update global validator metrics"""
        self.metrics['validations_performed'] += 1
        self.metrics['total_validation_time'] += result.validation_time
        self.metrics['average_validation_time'] = (
            self.metrics['total_validation_time'] / self.metrics['validations_performed']
        )
        
        if result.is_valid:
            self.metrics['modules_accepted'] += 1
        else:
            self.metrics['modules_rejected'] += 1
            
        self.metrics['threats_detected'] += len(result.threats_detected)

    def get_detailed_report(self, bytecode: bytes) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        result = self.validate_bytecode(bytecode)
        
        return {
            'validation_result': result.is_valid,
            'bytecode_hash': result.bytecode_hash,
            'validation_time': result.validation_time,
            'threat_summary': {
                'total': len(result.threats_detected),
                'critical': len([t for t in result.threats_detected 
                               if t['threat_level'] == SecurityThreatLevel.CRITICAL]),
                'high': len([t for t in result.threats_detected 
                           if t['threat_level'] == SecurityThreatLevel.HIGH]),
                'medium': len([t for t in result.threats_detected 
                             if t['threat_level'] == SecurityThreatLevel.MEDIUM]),
                'low': len([t for t in result.threats_detected 
                          if t['threat_level'] == SecurityThreatLevel.LOW]),
            },
            'threats_detected': result.threats_detected,
            'warnings': result.warnings,
            'errors': result.errors,
            'metrics': result.metrics,
            'validator_metrics': self.metrics.copy()
        }

    def add_custom_pattern(self, pattern: bytes, threat_level: SecurityThreatLevel,
                          description: str = "") -> None:
        """Add custom detection pattern"""
        self.malicious_patterns[pattern] = threat_level
        logger.info(f"Added custom pattern: {description or pattern.hex()}")

    def remove_pattern(self, pattern: bytes) -> bool:
        """Remove detection pattern"""
        if pattern in self.malicious_patterns:
            del self.malicious_patterns[pattern]
            logger.info(f"Removed pattern: {pattern.hex()}")
            return True
        return False

    def clear_cache(self) -> None:
        """Clear validation cache"""
        self.validation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Validation cache cleared")

    def get_validator_stats(self) -> Dict[str, Any]:
        """Get validator statistics"""
        return {
            **self.metrics,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_efficiency': self.cache_hits / (self.cache_hits + self.cache_misses) 
                              if (self.cache_hits + self.cache_misses) > 0 else 0,
            'malicious_patterns_count': len(self.malicious_patterns),
            'known_vulnerabilities_count': len(self.known_vulnerabilities),
            'validation_config': {
                'max_module_size': self.config.max_module_size,
                'validation_level': self.config.validation_level.value,
                'max_memory_pages': self.config.max_memory_pages
            }
        }

# Import time at the end to avoid circular imports
import time