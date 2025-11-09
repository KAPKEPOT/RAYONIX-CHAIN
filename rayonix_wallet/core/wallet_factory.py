#rayonix_wallet/core/wallet_factory.py
import logging
from typing import Optional, Tuple
from pathlib import Path

from rayonix_wallet.core.wallet import RayonixWallet
from config.config_manager import ConfigManager
from rayonix_wallet.core.wallet_types import WalletType, AddressType

logger = logging.getLogger(__name__)

class WalletFactory:
    """Factory for creating and managing wallet instances"""
    
    @staticmethod
    def create_new_wallet(
        wallet_type: WalletType = WalletType.HD,
        network: str = "mainnet",
        address_type: AddressType = AddressType.RAYONIX,
        config: Optional[WalletConfig] = None
    ) -> Tuple[RayonixWallet, str]:
        """
    Create a new wallet and return both wallet instance and mnemonic
    """
        try:
            # Create wallet configuration
            if config is None:
                config = WalletConfig(
                    wallet_type=wallet_type,
                    network=network,
                    address_type=address_type
                )
                
                # Ensure coin_type is set based on network
                if network == "mainnet":
                	config.coin_type = "1180"
                else:
                	config.coin_type = "1"
            
            # Create wallet instance
            wallet = RayonixWallet(config)
            
            # Generate and initialize with new mnemonic
            mnemonic = wallet.initialize_new_wallet()
            
            logger.info(f"Created new {wallet_type.name} wallet for {network}")
            return wallet, mnemonic
            
        except Exception as e:
            logger.error(f"Failed to create new wallet: {e}")
            raise
    
    @staticmethod
    def create_wallet_from_mnemonic(
        mnemonic: str,
        passphrase: str = "",
        wallet_type: WalletType = WalletType.HD,
        network: str = "mainnet",
        address_type: AddressType = AddressType.RAYONIX
    ) -> RayonixWallet:
        """
        Create wallet from existing mnemonic phrase
        
        Args:
            mnemonic: BIP39 mnemonic phrase
            passphrase: Optional passphrase
            wallet_type: Wallet type
            network: Network type
            address_type: Address format type
            
        Returns:
            Wallet instance
        """
        try:
            config = WalletConfig(
                wallet_type=wallet_type,
                network=network,
                address_type=address_type
            )
            
            wallet = RayonixWallet(config)
            
            if wallet.create_from_mnemonic(mnemonic, passphrase):
                logger.info(f"Successfully created wallet from mnemonic for {network}")
                return wallet
            else:
                raise ValueError("Failed to create wallet from mnemonic")
                
        except Exception as e:
            logger.error(f"Failed to create wallet from mnemonic: {e}")
            raise
    
    @staticmethod
    def create_wallet_from_private_key(
        private_key: str,
        wallet_type: WalletType = WalletType.NON_HD,
        network: str = "mainnet",
        address_type: AddressType = AddressType.RAYONIX
    ) -> RayonixWallet:
        """
        Create wallet from private key
        
        Args:
            private_key: Private key in hex format
            wallet_type: Wallet type
            network: Network type
            address_type: Address format type
            
        Returns:
            Wallet instance
        """
        try:
            config = WalletConfig(
                wallet_type=wallet_type,
                network=network,
                address_type=address_type
            )
            
            wallet = RayonixWallet(config)
            
            if wallet.create_from_private_key(private_key, wallet_type):
                logger.info(f"Successfully created wallet from private key for {network}")
                return wallet
            else:
                raise ValueError("Failed to create wallet from private key")
                
        except Exception as e:
            logger.error(f"Failed to create wallet from private key: {e}")
            raise
    
    @staticmethod
    def load_wallet_from_file(
        wallet_path: str,
        passphrase: str = ""
    ) -> RayonixWallet:
        """
        Load wallet from encrypted file
        
        Args:
            wallet_path: Path to wallet file
            passphrase: Decryption passphrase
            
        Returns:
            Wallet instance
        """
        try:
            wallet = RayonixWallet()
            
            if wallet.restore(wallet_path, passphrase):
                logger.info(f"Successfully loaded wallet from {wallet_path}")
                return wallet
            else:
                raise ValueError("Failed to load wallet from file")
                
        except Exception as e:
            logger.error(f"Failed to load wallet from file: {e}")
            raise

# Backward compatibility function
def create_new_wallet(
    wallet_type: WalletType = WalletType.HD,
    network: str = "mainnet", 
    address_type: AddressType = AddressType.RAYONIX
) -> Tuple[RayonixWallet, str]:
    """
    Backward compatibility function - creates new wallet and returns instance with mnemonic
    """
    return WalletFactory.create_new_wallet(wallet_type, network, address_type)