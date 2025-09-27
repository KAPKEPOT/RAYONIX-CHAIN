# blockchain/utils/config_utils.py

def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Safely get value from config whether it's a dict or dataclass"""
    if hasattr(config, '__dataclass_fields__'):
        # It's a dataclass
        return getattr(config, key, default)
    elif isinstance(config, dict):
        # It's a dictionary
        return config.get(key, default)
    else:
        return default