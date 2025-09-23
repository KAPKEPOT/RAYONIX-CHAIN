# smart_contract/exceptions/security_errors.py
"""
Security-related exceptions for smart contract execution
"""

class SecurityError(Exception):
    """Base class for all security-related exceptions"""
    pass
    
class SecurityViolationError(SecurityError):
    """General security violation detected"""
    
    def __init__(self, message: str, violation_type: str = "GENERAL"):
        self.violation_type = violation_type
        super().__init__(f"SecurityViolation[{violation_type}]: {message}")

class RateLimitExceededError(SecurityViolationError):
    """Rate limit exceeded"""
    
    def __init__(self, address: str, operation: str, limit: int, current: int):
        message = f"Rate limit exceeded for {address} operation {operation}: {current}/{limit}"
        super().__init__(message, "RATE_LIMIT")

class BlacklistedAddressError(SecurityViolationError):
    """Blacklisted address attempted access"""
    
    def __init__(self, address: str, reason: str = "Unknown"):
        message = f"Blacklisted address {address} attempted access. Reason: {reason}"
        super().__init__(message, "BLACKLISTED")    

class ContractSecurityError(SecurityError):
    """Security violation detected in contract execution"""
    
    def __init__(self, message: str, contract_address: str = None, severity: str = "HIGH"):
        self.contract_address = contract_address
        self.severity = severity
        super().__init__(f"SecurityError[{severity}] {contract_address or 'Unknown'}: {message}")

class AccessControlError(ContractSecurityError):
    """Unauthorized access attempt"""
    
    def __init__(self, operation: str, caller: str, required_role: str = None):
        message = f"Unauthorized access to {operation} by {caller}"
        if required_role:
            message += f". Required role: {required_role}"
        super().__init__(message, caller, "HIGH")

class ReentrancyAttackError(ContractSecurityError):
    """Detected reentrancy attack attempt"""
    
    def __init__(self, contract_address: str, attacker: str, method: str):
        message = f"Reentrancy attack detected on {method} by {attacker}"
        super().__init__(message, contract_address, "CRITICAL")

class IntegerOverflowError(ContractSecurityError):
    """Integer overflow/underflow detected"""
    
    def __init__(self, contract_address: str, operation: str, values: tuple):
        message = f"Integer overflow in {operation} with values {values}"
        super().__init__(message, contract_address, "HIGH")

class UncheckedCallError(ContractSecurityError):
    """Dangerous unchecked low-level call"""
    
    def __init__(self, contract_address: str, target: str, value: int = 0):
        message = f"Unchecked call to {target} with value {value}"
        super().__init__(message, contract_address, "MEDIUM")

class GasExhaustionAttackError(ContractSecurityError):
    """Potential gas exhaustion attack detected"""
    
    def __init__(self, contract_address: str, attacker: str, gas_used: int):
        message = f"Gas exhaustion attack by {attacker}, used {gas_used} gas"
        super().__init__(message, contract_address, "HIGH")

class TimestampDependencyError(ContractSecurityError):
    """Dangerous timestamp dependency detected"""
    
    def __init__(self, contract_address: str, operation: str):
        message = f"Timestamp dependency in {operation}"
        super().__init__(message, contract_address, "MEDIUM")

class BlockhashDependencyError(ContractSecurityError):
    """Dangerous blockhash dependency detected"""
    
    def __init__(self, contract_address: str, operation: str):
        message = f"Blockhash dependency in {operation}"
        super().__init__(message, contract_address, "MEDIUM")

class OracleManipulationError(ContractSecurityError):
    """Potential oracle manipulation detected"""
    
    def __init__(self, contract_address: str, oracle_source: str, manipulated_value: any):
        message = f"Oracle manipulation from {oracle_source}, value: {manipulated_value}"
        super().__init__(message, contract_address, "HIGH")

class FrontRunningError(ContractSecurityError):
    """Front-running attack detected"""
    
    def __init__(self, contract_address: str, attacker: str, victim: str, amount: int):
        message = f"Front-running attack by {attacker} against {victim} for {amount}"
        super().__init__(message, contract_address, "MEDIUM")

class DenialOfServiceError(ContractSecurityError):
    """Denial of service attempt detected"""
    
    def __init__(self, contract_address: str, attacker: str, method: str):
        message = f"DoS attempt on {method} by {attacker}"
        super().__init__(message, contract_address, "HIGH")

class SignatureVerificationError(ContractSecurityError):
    """Signature verification failed"""
    
    def __init__(self, contract_address: str, signer: str, reason: str):
        message = f"Signature verification failed for {signer}: {reason}"
        super().__init__(message, contract_address, "HIGH")

class MaliciousBytecodeError(ContractSecurityError):
    """Malicious bytecode detected"""
    
    def __init__(self, contract_address: str, issue: str):
        message = f"Malicious bytecode: {issue}"
        super().__init__(message, contract_address, "CRITICAL")

class UnauthorizedUpgradeError(ContractSecurityError):
    """Unauthorized contract upgrade attempt"""
    
    def __init__(self, contract_address: str, caller: str, new_impl: str):
        message = f"Unauthorized upgrade attempt by {caller} to {new_impl}"
        super().__init__(message, contract_address, "HIGH")

class StorageCollisionError(ContractSecurityError):
    """Storage collision detected in proxy pattern"""
    
    def __init__(self, contract_address: str, slot: str, collision_with: str):
        message = f"Storage collision at slot {slot} with {collision_with}"
        super().__init__(message, contract_address, "HIGH")

class FunctionVisibilityError(ContractSecurityError):
    """Incorrect function visibility"""
    
    def __init__(self, contract_address: str, function: str, expected: str, actual: str):
        message = f"Function {function} visibility {actual}, expected {expected}"
        super().__init__(message, contract_address, "MEDIUM")

class EventSpoofingError(ContractSecurityError):
    """Event spoofing attempt detected"""
    
    def __init__(self, contract_address: str, event: str, spoofed_data: dict):
        message = f"Event spoofing for {event}: {spoofed_data}"
        super().__init__(message, contract_address, "MEDIUM")

class PhishingAttemptError(ContractSecurityError):
    """Phishing attempt detected"""
    
    def __init__(self, contract_address: str, malicious_contract: str, technique: str):
        message = f"Phishing attempt via {technique} from {malicious_contract}"
        super().__init__(message, contract_address, "HIGH")

class TokenTheftAttemptError(ContractSecurityError):
    """Token theft attempt detected"""
    
    def __init__(self, contract_address: str, attacker: str, target: str, amount: int):
        message = f"Token theft attempt by {attacker} from {target} for {amount}"
        super().__init__(message, contract_address, "CRITICAL")

class FlashLoanAttackError(ContractSecurityError):
    """Flash loan attack detected"""
    
    def __init__(self, contract_address: str, attacker: str, loan_amount: int, profit: int):
        message = f"Flash loan attack by {attacker}, loan: {loan_amount}, profit: {profit}"
        super().__init__(message, contract_address, "HIGH")

class PriceManipulationError(ContractSecurityError):
    """Price manipulation attack detected"""
    
    def __init__(self, contract_address: str, pair: str, manipulation_amount: float):
        message = f"Price manipulation on {pair} by {manipulation_amount}"
        super().__init__(message, contract_address, "HIGH")

class GovernanceAttackError(ContractSecurityError):
    """Governance attack detected"""
    
    def __init__(self, contract_address: str, attacker: str, voting_power: int):
        message = f"Governance attack by {attacker} with {voting_power} power"
        super().__init__(message, contract_address, "HIGH")

class TimeLockBypassError(ContractSecurityError):
    """Timelock bypass attempt detected"""
    
    def __init__(self, contract_address: str, attacker: str, time_remaining: int):
        message = f"Timelock bypass attempt by {attacker}, {time_remaining}s remaining"
        super().__init__(message, contract_address, "HIGH")

class WhitelistBypassError(ContractSecurityError):
    """Whitelist bypass attempt detected"""
    
    def __init__(self, contract_address: str, attacker: str, method: str):
        message = f"Whitelist bypass attempt on {method} by {attacker}"
        super().__init__(message, contract_address, "MEDIUM")

class SandboxEscapeError(ContractSecurityError):
    """WASM sandbox escape attempt detected"""
    
    def __init__(self, contract_address: str, attempt_details: str):
        message = f"Sandbox escape attempt: {attempt_details}"
        super().__init__(message, contract_address, "CRITICAL")

class MemoryCorruptionError(ContractSecurityError):
    """Memory corruption attempt detected"""
    
    def __init__(self, contract_address: str, operation: str, target_address: int):
        message = f"Memory corruption in {operation} at address {target_address}"
        super().__init__(message, contract_address, "CRITICAL")

class InfiniteLoopError(ContractSecurityError):
    """Potential infinite loop detected"""
    
    def __init__(self, contract_address: str, loop_condition: str):
        message = f"Potential infinite loop with condition: {loop_condition}"
        super().__init__(message, contract_address, "HIGH")

class ResourceExhaustionError(ContractSecurityError):
    """Resource exhaustion attempt detected"""
    
    def __init__(self, contract_address: str, resource: str, usage: int, limit: int):
        message = f"Resource exhaustion: {resource} usage {usage}/{limit}"
        super().__init__(message, contract_address, "HIGH")

class CryptographicWeaknessError(ContractSecurityError):
    """Cryptographic weakness detected"""
    
    def __init__(self, contract_address: str, algorithm: str, issue: str):
        message = f"Cryptographic weakness in {algorithm}: {issue}"
        super().__init__(message, contract_address, "HIGH")

class DomainValidationError(SecurityError):
    """Domain validation failed"""
    pass

class IPValidationError(SecurityError):
    """IP address validation failed"""
    pass

class InputValidationError(SecurityError):
    """Input validation failed"""
    
    def __init__(self, field: str, value: any, constraint: str):
        message = f"Input validation failed for {field}={value}, constraint: {constraint}"
        super().__init__(message)

class BehavioralAnomalyError(SecurityError):
    """Behavioral anomaly detected"""
    
    def __init__(self, contract_address: str, behavior: str, confidence: float):
        message = f"Behavioral anomaly: {behavior} (confidence: {confidence:.2f})"
        super().__init__(message)

class ThreatIntelligenceError(SecurityError):
    """Threat intelligence match found"""
    
    def __init__(self, contract_address: str, threat_type: str, source: str):
        message = f"Threat intelligence match: {threat_type} from {source}"
        super().__init__(message)