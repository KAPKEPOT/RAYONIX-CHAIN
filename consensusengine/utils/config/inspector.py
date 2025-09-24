# utils/config/inspector.py
import inspect
from consensusengine.utils.config.settings import ConsensusConfig

def inspect_consensus_config():
    """Inspect the actual ConsensusConfig class structure"""
    print("=== ConsensusConfig Inspection ===")
    print(f"Module: {ConsensusConfig.__module__}")
    print(f"Class: {ConsensusConfig.__name__}")
    
    # Check if it's a dataclass
    if hasattr(ConsensusConfig, '__dataclass_fields__'):
        print("Type: Dataclass")
        print("Fields:")
        for field_name, field in ConsensusConfig.__dataclass_fields__.items():
            print(f"  - {field_name}: {field.type}")
    else:
        print("Type: Regular class")
        # Check __init__ signature
        init_signature = inspect.signature(ConsensusConfig.__init__)
        print("Init parameters:")
        for param_name, param in init_signature.parameters.items():
            if param_name != 'self':
                print(f"  - {param_name}: {param.default}")

if __name__ == "__main__":
    inspect_consensus_config()