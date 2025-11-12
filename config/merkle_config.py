@dataclass
class MerkleDatabaseConfig:
    """Configuration for Merkle tree integrity protection"""
    enabled: bool = True
    merkle_tree_depth: int = 256
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    double_hash: bool = True
    verify_on_read: bool = True
    verify_on_write: bool = True
    auto_recover: bool = True
    store_proofs: bool = False
    proof_format: ProofFormat = ProofFormat.BINARY
    integrity_check_interval: int = 3600  # seconds