"""
Configuration validation for consensus system
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import is_dataclass, fields
from pathlib import Path
import ipaddress
import logging

from .settings import Settings, Environment
from ..exceptions import ConsensusError

logger = logging.getLogger('consensus.config')

class ConfigValidator:
    """Configuration validator"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_settings(self, settings: Settings) -> bool:
        """Validate complete settings object"""
        self.errors.clear()
        self.warnings.clear()
        
        try:
            # Validate basic settings
            self._validate_environment(settings.environment)
            self._validate_node_id(settings.node_id)
            self._validate_data_dir(settings.data_dir)
            
            # Validate nested configurations
            self._validate_database_config(settings.database)
            self._validate_network_config(settings.network)
            self._validate_consensus_config(settings.consensus)
            self._validate_crypto_config(settings.crypto)
            self._validate_metrics_config(settings.metrics)
            self._validate_api_config(settings.api)
            self._validate_logging_config(settings.logging)
            
            # Cross-field validation
            self._validate_cross_field(settings)
            
            if self.errors:
                logger.error(f"Configuration validation failed: {self.errors}")
                return False
            
            if self.warnings:
                logger.warning(f"Configuration warnings: {self.warnings}")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Validation error: {e}")
            return False
    
    def _validate_environment(self, environment: Environment) -> None:
        """Validate environment setting"""
        if not isinstance(environment, Environment):
            self.errors.append("environment must be a valid Environment enum")
    
    def _validate_node_id(self, node_id: str) -> None:
        """Validate node ID"""
        if not node_id or not isinstance(node_id, str):
            self.errors.append("node_id must be a non-empty string")
            return
        
        if len(node_id) > 50:
            self.errors.append("node_id must be 50 characters or less")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', node_id):
            self.errors.append("node_id can only contain letters, numbers, underscores, and hyphens")
    
    def _validate_data_dir(self, data_dir: str) -> None:
        """Validate data directory"""
        if not data_dir or not isinstance(data_dir, str):
            self.errors.append("data_dir must be a non-empty string")
            return
        
        path = Path(data_dir)
        if not path.parent.exists():
            self.errors.append(f"Parent directory of data_dir does not exist: {path.parent}")
    
    def _validate_database_config(self, config: Any) -> None:
        """Validate database configuration"""
        if not is_dataclass(config):
            self.errors.append("database config must be a dataclass")
            return
        
        if config.max_open_files < 100:
            self.warnings.append("max_open_files is very low, may cause performance issues")
        
        if config.write_buffer_size < 1024 * 1024:  # 1MB
            self.warnings.append("write_buffer_size is very small, may cause performance issues")
    
    def _validate_network_config(self, config: Any) -> None:
        """Validate network configuration"""
        if not is_dataclass(config):
            self.errors.append("network config must be a dataclass")
            return
        
        # Validate host
        try:
            ipaddress.ip_address(config.host)
        except ValueError:
            self.errors.append(f"Invalid network host: {config.host}")
        
        # Validate port
        if config.port < 1024 or config.port > 65535:
            self.errors.append(f"Network port must be between 1024 and 65535: {config.port}")
        
        # Validate max message size
        if config.max_message_size > 100 * 1024 * 1024:  # 100MB
            self.warnings.append("max_message_size is very large, may cause memory issues")
    
    def _validate_consensus_config(self, config: Any) -> None:
        """Validate consensus configuration"""
        if not is_dataclass(config):
            self.errors.append("consensus config must be a dataclass")
            return
        
        # Validate timeouts
        if config.timeout_propose < 1000:
            self.warnings.append("timeout_propose is very short, may cause frequent timeouts")
        
        if config.timeout_prevote < 500:
            self.warnings.append("timeout_prevote is very short, may cause frequent timeouts")
        
        # Validate stake parameters
        if config.min_stake < 1:
            self.errors.append("min_stake must be at least 1")
        
        if config.max_validators < 1:
            self.errors.append("max_validators must be at least 1")
        
        if not 0 <= config.slash_percentage <= 1:
            self.errors.append("slash_percentage must be between 0 and 1")
    
    def _validate_crypto_config(self, config: Any) -> None:
        """Validate cryptography configuration"""
        if not is_dataclass(config):
            self.errors.append("crypto config must be a dataclass")
            return
        
        valid_algorithms = {"secp256k1", "rsa", "ed25519"}
        if config.key_algorithm not in valid_algorithms:
            self.errors.append(f"key_algorithm must be one of {valid_algorithms}")
        
        valid_hashes = {"sha256", "sha3_256", "blake2b"}
        if config.hash_algorithm not in valid_hashes:
            self.errors.append(f"hash_algorithm must be one of {valid_hashes}")
    
    def _validate_metrics_config(self, config: Any) -> None:
        """Validate metrics configuration"""
        if not is_dataclass(config):
            self.errors.append("metrics config must be a dataclass")
            return
        
        if config.port < 1024 or config.port > 65535:
            self.errors.append(f"Metrics port must be between 1024 and 65535: {config.port}")
    
    def _validate_api_config(self, config: Any) -> None:
        """Validate API configuration"""
        if not is_dataclass(config):
            self.errors.append("api config must be a dataclass")
            return
        
        if config.port < 1024 or config.port > 65535:
            self.errors.append(f"API port must be between 1024 and 65535: {config.port}")
        
        if config.timeout < 1:
            self.errors.append("API timeout must be at least 1 second")
    
    def _validate_logging_config(self, config: Any) -> None:
        """Validate logging configuration"""
        if not is_dataclass(config):
            self.errors.append("logging config must be a dataclass")
            return
        
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if config.level not in valid_levels:
            self.errors.append(f"log level must be one of {valid_levels}")
        
        if config.file and len(config.file) > 255:
            self.errors.append("log file path too long")
    
    def _validate_cross_field(self, settings: Settings) -> None:
        """Cross-field validation"""
        # Check if API and metrics ports conflict
        if settings.api.enabled and settings.metrics.enabled:
            if settings.api.port == settings.metrics.port:
                self.errors.append("API and metrics cannot use the same port")
        
        # Check if network and API ports conflict
        if settings.network.port == settings.api.port:
            self.errors.append("Network and API cannot use the same port")
        
        # Production-specific validations
        if settings.is_production():
            if settings.consensus.timeout_propose > 5000:
                self.warnings.append("timeout_propose is long for production")
            
            if not settings.network.peer_discovery:
                self.warnings.append("peer_discovery is disabled in production")
    
    def get_errors(self) -> List[str]:
        """Get validation errors"""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get validation warnings"""
        return self.warnings.copy()
    
    def format_validation_report(self) -> str:
        """Format validation results as a report"""
        report = []
        
        if self.errors:
            report.append("VALIDATION ERRORS:")
            for error in self.errors:
                report.append(f"  ✗ {error}")
        
        if self.warnings:
            report.append("\nVALIDATION WARNINGS:")
            for warning in self.warnings:
                report.append(f"  ⚠ {warning}")
        
        if not self.errors and not self.warnings:
            report.append("✓ Configuration is valid")
        
        return "\n".join(report)

def validate_config_file(config_path: str) -> bool:
    """Validate configuration file"""
    try:
        settings = Settings.from_file(config_path)
        validator = ConfigValidator()
        return validator.validate_settings(settings)
    except Exception as e:
        logger.error(f"Config file validation failed: {e}")
        return False

def create_default_config(config_path: str, environment: Environment = Environment.DEVELOPMENT) -> None:
    """Create default configuration file"""
    try:
        settings = Settings(environment=environment)
        settings.save_to_file(config_path)
        logger.info(f"Created default config at {config_path}")
    except Exception as e:
        logger.error(f"Failed to create default config: {e}")
        raise