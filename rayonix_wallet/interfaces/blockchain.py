from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BlockchainInterface(ABC):
    """Abstract base class for blockchain interfaces"""
    
    @abstractmethod
    def get_block_height(self) -> int:
        """Get current blockchain height"""
        pass
    
    @abstractmethod
    def get_address_balance(self, address: str) -> Dict[str, int]:
        """Get balance for address"""
        pass
    
    @abstractmethod
    def get_address_transactions(self, address: str) -> List[Dict]:
        """Get transactions for address"""
        pass
    
    @abstractmethod
    def get_transaction(self, txid: str) -> Optional[Dict]:
        """Get transaction details"""
        pass
    
    @abstractmethod
    def broadcast_transaction(self, raw_tx: str) -> Optional[str]:
        """Broadcast raw transaction"""
        pass
    
    @abstractmethod
    def get_fee_estimate(self, priority: str = "medium") -> int:
        """Get transaction fee estimate"""
        pass
    
    @abstractmethod
    def get_utxos(self, address: str) -> List[Dict]:
        """Get UTXOs for address"""
        pass
    
    @abstractmethod
    def subscribe_address(self, address: str) -> bool:
        """Subscribe to address updates"""
        pass
    
    @abstractmethod
    def unsubscribe_address(self, address: str) -> bool:
        """Unsubscribe from address updates"""
        pass

class RayonixBlockchainInterface(BlockchainInterface):
    """Rayonix-specific blockchain interface"""
    
    def __init__(self, network: str = "mainnet", api_url: Optional[str] = None):
        self.network = network
        self.api_url = api_url or self._get_default_api_url()
        self.session = None
    
    def _get_default_api_url(self) -> str:
        """Get default API URL based on network"""
        if self.network == "mainnet":
            return "https://api.rayonix.org"
        else:
            return "https://api-testnet.rayonix.org"
    
    def get_block_height(self) -> int:
        """Get current blockchain height"""
        try:
            response = self._make_api_call("getblockcount")
            return response.get('result', 0)
        except Exception as e:
            logger.error(f"Failed to get block height: {e}")
            return 0
    
    def get_address_balance(self, address: str) -> Dict[str, int]:
        """Get balance for address"""
        try:
            response = self._make_api_call("getaddressbalance", {"address": address})
            return {
                'balance': response.get('balance', 0),
                'received': response.get('total_received', 0),
                'sent': response.get('total_sent', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get address balance: {e}")
            return {'balance': 0, 'received': 0, 'sent': 0}
    
    def get_address_transactions(self, address: str) -> List[Dict]:
        """Get transactions for address"""
        try:
            response = self._make_api_call("getaddresstxs", {"address": address})
            return response.get('transactions', [])
        except Exception as e:
            logger.error(f"Failed to get address transactions: {e}")
            return []
    
    def get_transaction(self, txid: str) -> Optional[Dict]:
        """Get transaction details"""
        try:
            response = self._make_api_call("gettransaction", {"txid": txid})
            return response.get('result')
        except Exception as e:
            logger.error(f"Failed to get transaction: {e}")
            return None
    
    def broadcast_transaction(self, raw_tx: str) -> Optional[str]:
        """Broadcast raw transaction"""
        try:
            response = self._make_api_call("sendrawtransaction", {"hexstring": raw_tx})
            return response.get('result')
        except Exception as e:
            logger.error(f"Failed to broadcast transaction: {e}")
            return None
    
    def get_fee_estimate(self, priority: str = "medium") -> int:
        """Get transaction fee estimate"""
        try:
            response = self._make_api_call("estimatesmartfee", {"conf_target": 6})
            return int(response.get('result', {}).get('feerate', 0.0001) * 100000000)
        except Exception as e:
            logger.error(f"Failed to get fee estimate: {e}")
            return 2  # Default fallback
    
    def get_utxos(self, address: str) -> List[Dict]:
        """Get UTXOs for address"""
        try:
            response = self._make_api_call("getaddressutxos", {"address": address})
            return response.get('result', [])
        except Exception as e:
            logger.error(f"Failed to get UTXOs: {e}")
            return []
    
    def subscribe_address(self, address: str) -> bool:
        """Subscribe to address updates"""
        try:
            response = self._make_api_call("subscribe", {"address": address})
            return response.get('result', False)
        except Exception as e:
            logger.error(f"Failed to subscribe to address: {e}")
            return False
    
    def unsubscribe_address(self, address: str) -> bool:
        """Unsubscribe from address updates"""
        try:
            response = self._make_api_call("unsubscribe", {"address": address})
            return response.get('result', False)
        except Exception as e:
            logger.error(f"Failed to unsubscribe from address: {e}")
            return False
    
    def _make_api_call(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Make API call to blockchain node"""
        import requests
        import json
        
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'User-Agent': 'RayonixWallet/1.0'
            })
        
        payload = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': method,
            'params': params or {}
        }
        
        try:
            response = self.session.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            raise