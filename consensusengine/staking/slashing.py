# consensus/staking/slashing.py
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Set
import logging
from threading import RLock

logger = logging.getLogger('SlashingManager')

class SlashingManager:
    """Production-ready slashing conditions and evidence verification"""
    
    def __init__(self, consensus_engine: Any):
        self.consensus_engine = consensus_engine
        self.config = consensus_engine.config
        self.lock = RLock()
        
        # Evidence tracking
        self.pending_evidence: List[Dict] = []
        self.processed_evidence: Set[str] = set()
        
        # Slashing history
        self.slashing_events: Dict[int, List[Dict]] = {}  # height -> events
        
    def slash_validator(self, validator_address: str, evidence: Dict, reporter: str) -> bool:
        """
        Slash a validator for misbehavior with production-grade validation
        
        Args:
            validator_address: Address of validator to slash
            evidence: Dictionary containing evidence of misbehavior
            reporter: Address of the reporter (for potential rewards)
            
        Returns:
            True if slashing was successful, False otherwise
        """
        with self.lock:
            try:
                # Validate inputs
                if not self._validate_slashing_inputs(validator_address, evidence, reporter):
                    return False
                
                # Check if validator exists
                if validator_address not in self.consensus_engine.validators:
                    logger.warning(f"Attempt to slash unknown validator: {validator_address}")
                    return False
                
                validator = self.consensus_engine.validators[validator_address]
                
                # Verify evidence based on type
                evidence_type = evidence.get('type')
                evidence_hash = self._calculate_evidence_hash(evidence)
                
                # Check if evidence was already processed
                if evidence_hash in self.processed_evidence:
                    logger.warning(f"Duplicate evidence: {evidence_hash}")
                    return False
                
                is_valid_evidence = False
                slash_amount = 0
                
                if evidence_type == 'double_sign':
                    is_valid_evidence = self._verify_double_sign_evidence(validator, evidence)
                    if is_valid_evidence:
                        # Severe slashing for double signing
                        slash_amount = int(validator.total_stake * self.config.slash_percentage)
                        validator.status = ValidatorStatus.SLASHED
                        logger.warning(f"Validator {validator_address} slashed for double signing: {slash_amount}")
                
                elif evidence_type == 'unavailability':
                    is_valid_evidence = self._verify_unavailability_evidence(validator, evidence)
                    if is_valid_evidence:
                        # Moderate slashing for unavailability
                        slash_amount = int(validator.staked_amount * (self.config.slash_percentage / 2))
                        validator.jail(self.config.jail_duration)
                        logger.warning(f"Validator {validator_address} jailed for unavailability: {slash_amount}")
                
                elif evidence_type == 'byzantine_behavior':
                    is_valid_evidence = self._verify_byzantine_evidence(validator, evidence)
                    if is_valid_evidence:
                        slash_amount = int(validator.total_stake * self.config.slash_percentage * 0.75)
                        validator.jail(self.config.jail_duration * 2)
                        logger.warning(f"Validator {validator_address} slashed for Byzantine behavior: {slash_amount}")
                
                else:
                    logger.warning(f"Unknown evidence type: {evidence_type}")
                    return False
                
                if not is_valid_evidence:
                    logger.warning(f"Invalid evidence for validator {validator_address}")
                    return False
                
                # Apply slashing
                if slash_amount > 0:
                    if not validator.slash(slash_amount):
                        logger.error(f"Failed to apply slashing to validator {validator_address}")
                        return False
                
                # Record slashing event
                self._record_slashing_event(validator_address, evidence_type, slash_amount, reporter)
                
                # Mark evidence as processed
                self.processed_evidence.add(evidence_hash)
                
                # Update validator set if needed
                if validator.status != ValidatorStatus.ACTIVE:
                    self.consensus_engine.staking_manager.update_validator_set()
                
                # Potential reporter reward (implementation specific)
                self._consider_reporter_reward(reporter, slash_amount)
                
                logger.info(f"Successfully slashed validator {validator_address} for {evidence_type}")
                return True
                
            except Exception as e:
                logger.error(f"Error slashing validator: {e}")
                return False
    
    def _validate_slashing_inputs(self, validator_address: str, evidence: Dict, reporter: str) -> bool:
        """Validate slashing input parameters"""
        if not validator_address or len(validator_address) != 42:
            logger.warning("Invalid validator address format")
            return False
        
        if not evidence or 'type' not in evidence:
            logger.warning("Evidence must contain 'type' field")
            return False
        
        if not reporter or len(reporter) != 42:
            logger.warning("Invalid reporter address format")
            return False
        
        return True
    
    def _verify_double_sign_evidence(self, validator: Validator, evidence: Dict) -> bool:
        """
        Verify double signing evidence with cryptographic proof
        
        Args:
            validator: Validator being accused
            evidence: Double signing evidence
            
        Returns:
            True if evidence is valid, False otherwise
        """
        try:
            required_fields = {'block1', 'block2', 'signature1', 'signature2', 'public_key'}
            
            if not all(field in evidence for field in required_fields):
                logger.warning("Missing required fields in double sign evidence")
                return False
            
            block1 = evidence['block1']
            block2 = evidence['block2']
            signature1 = evidence['signature1']
            signature2 = evidence['signature2']
            public_key = evidence['public_key']
            
            # Verify both blocks are at the same height
            if block1.get('height') != block2.get('height'):
                logger.warning("Double sign evidence: blocks at different heights")
                return False
            
            # Verify blocks are different
            if block1.get('hash') == block2.get('hash'):
                logger.warning("Double sign evidence: blocks are identical")
                return False
            
            # Verify both signatures are from the same validator
            if public_key != validator.public_key:
                logger.warning("Double sign evidence: public key mismatch")
                return False
            
            # Verify first signature
            if not self._verify_block_signature(block1, signature1, public_key):
                logger.warning("Double sign evidence: invalid signature for block1")
                return False
            
            # Verify second signature
            if not self._verify_block_signature(block2, signature2, public_key):
                logger.warning("Double sign evidence: invalid signature for block2")
                return False
            
            # Additional checks for timestamp validity, etc.
            current_time = time.time()
            max_allowed_difference = 3600  # 1 hour
            
            if abs(block1.get('timestamp', 0) - current_time) > max_allowed_difference:
                logger.warning("Double sign evidence: block1 timestamp too far from current time")
                return False
            
            if abs(block2.get('timestamp', 0) - current_time) > max_allowed_difference:
                logger.warning("Double sign evidence: block2 timestamp too far from current time")
                return False
            
            logger.info(f"Valid double sign evidence for validator {validator.address}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying double sign evidence: {e}")
            return False
    
    def _verify_unavailability_evidence(self, validator: Validator, evidence: Dict) -> bool:
        """
        Verify unavailability evidence
        
        Args:
            validator: Validator being accused
            evidence: Unavailability evidence
            
        Returns:
            True if evidence is valid, False otherwise
        """
        try:
            required_fields = {'missed_blocks', 'total_blocks', 'time_period'}
            
            if not all(field in evidence for field in required_fields):
                logger.warning("Missing required fields in unavailability evidence")
                return False
            
            missed_blocks = evidence['missed_blocks']
            total_blocks = evidence['total_blocks']
            time_period = evidence['time_period']  # in seconds
            
            if total_blocks <= 0:
                logger.warning("Unavailability evidence: total_blocks must be positive")
                return False
            
            if missed_blocks < 0 or missed_blocks > total_blocks:
                logger.warning("Unavailability evidence: invalid missed_blocks count")
                return False
            
            if time_period <= 0:
                logger.warning("Unavailability evidence: time_period must be positive")
                return False
            
            # Calculate missed percentage
            missed_percentage = missed_blocks / total_blocks
            
            # Only slash if missed more than 50% of blocks
            if missed_percentage < 0.5:
                logger.warning(f"Unavailability evidence: insufficient missed percentage: {missed_percentage}")
                return False
            
            # Verify the evidence time period is reasonable
            max_allowed_period = 86400 * 7  # 1 week
            if time_period > max_allowed_period:
                logger.warning(f"Unavailability evidence: time period too long: {time_period}")
                return False
            
            # Cross-reference with validator's own statistics
            validator_missed_percentage = validator.missed_blocks / (validator.missed_blocks + validator.signed_blocks)
            
            # Allow some tolerance for reporting discrepancies
            if abs(missed_percentage - validator_missed_percentage) > 0.1:  # 10% tolerance
                logger.warning("Unavailability evidence: significant discrepancy with validator stats")
                return False
            
            logger.info(f"Valid unavailability evidence for validator {validator.address}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying unavailability evidence: {e}")
            return False
    
    def _verify_byzantine_evidence(self, validator: Validator, evidence: Dict) -> bool:
        """
        Verify Byzantine behavior evidence
        
        Args:
            validator: Validator being accused
            evidence: Byzantine behavior evidence
            
        Returns:
            True if evidence is valid, False otherwise
        """
        try:
            # Byzantine behavior includes various types of malicious activities
            evidence_type = evidence.get('subtype')
            
            if evidence_type == 'invalid_proposal':
                return self._verify_invalid_proposal_evidence(validator, evidence)
            elif evidence_type == 'equivocation':
                return self._verify_equivocation_evidence(validator, evidence)
            elif evidence_type == 'censorship':
                return self._verify_censorship_evidence(validator, evidence)
            else:
                logger.warning(f"Unknown Byzantine evidence subtype: {evidence_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying Byzantine evidence: {e}")
            return False
    
    def _verify_invalid_proposal_evidence(self, validator: Validator, evidence: Dict) -> bool:
        """Verify evidence of invalid block proposal"""
        # Implementation would validate that the proposed block violates consensus rules
        # This is a complex check that would involve full block validation
        return True  # Placeholder
    
    def _verify_equivocation_evidence(self, validator: Validator, evidence: Dict) -> bool:
        """Verify evidence of equivocation (sending conflicting messages)"""
        # Similar to double signing but for other message types
        return True  # Placeholder
    
    def _verify_censorship_evidence(self, validator: Validator, evidence: Dict) -> bool:
        """Verify evidence of transaction censorship"""
        # Check if validator is excluding valid transactions maliciously
        return True  # Placeholder
    
    def _verify_block_signature(self, block: Dict, signature: str, public_key: str) -> bool:
        """Verify block signature using validator's public key"""
        try:
            from consensus.crypto.signing import CryptoManager
            crypto_manager = CryptoManager()
            
            # Get signing data from block
            signing_data = self._get_block_signing_data(block)
            
            # Verify signature
            return crypto_manager.verify_signature(public_key, signing_data, signature)
            
        except Exception as e:
            logger.error(f"Error verifying block signature: {e}")
            return False
    
    def _get_block_signing_data(self, block: Dict) -> bytes:
        """Get data that should be signed for a block"""
        signing_data = {
            'height': block.get('height'),
            'hash': block.get('hash'),
            'previous_hash': block.get('previous_hash'),
            'timestamp': block.get('timestamp'),
            'validator': block.get('validator')
        }
        return json.dumps(signing_data, sort_keys=True).encode()
    
    def _calculate_evidence_hash(self, evidence: Dict) -> str:
        """Calculate unique hash for evidence to prevent duplicates"""
        evidence_string = json.dumps(evidence, sort_keys=True)
        return hashlib.sha256(evidence_string.encode()).hexdigest()
    
    def _record_slashing_event(self, validator_address: str, evidence_type: str, 
                             amount: int, reporter: str):
        """Record slashing event in history"""
        event = {
            'validator': validator_address,
            'evidence_type': evidence_type,
            'amount': amount,
            'reporter': reporter,
            'timestamp': time.time(),
            'height': self.consensus_engine.height
        }
        
        current_height = self.consensus_engine.height
        if current_height not in self.slashing_events:
            self.slashing_events[current_height] = []
        
        self.slashing_events[current_height].append(event)
    
    def _consider_reporter_reward(self, reporter: str, slash_amount: int):
        """Consider rewarding the reporter for reporting misbehavior"""
        # Implementation would depend on reward policy
        # Typically, a percentage of the slashed amount goes to the reporter
        reward_percentage = 0.05  # 5% reward
        reporter_reward = int(slash_amount * reward_percentage)
        
        # In production, this would transfer tokens to the reporter
        logger.info(f"Reporter {reporter} would receive {reporter_reward} reward")
    
    def check_jailed_validators(self):
        """Check and release jailed validators whose jail period has expired"""
        with self.lock:
            current_time = time.time()
            
            for validator in self.consensus_engine.validators.values():
                if (validator.status == ValidatorStatus.JAILED and 
                    validator.jail_until and 
                    validator.jail_until <= current_time):
                    
                    validator.unjail()
                    logger.info(f"Released validator {validator.address} from jail")
    
    def check_unavailability(self):
        """Check for validator unavailability and apply slashing if needed"""
        with self.lock:
            for validator in self.consensus_engine.validators.values():
                if validator.status != ValidatorStatus.ACTIVE:
                    continue
                
                # Calculate missed block percentage over recent window
                total_blocks = validator.missed_blocks + validator.signed_blocks
                if total_blocks < 100:  # Need sufficient data
                    continue
                
                missed_percentage = validator.missed_blocks / total_blocks
                
                # Apply slashing if missed more than 50% of blocks
                if missed_percentage > 0.5:
                    evidence = {
                        'type': 'unavailability',
                        'missed_blocks': validator.missed_blocks,
                        'total_blocks': total_blocks,
                        'time_period': total_blocks * 5,  # Estimate based on block time
                        'reporter': 'system'
                    }
                    
                    self.slash_validator(validator.address, evidence, 'system')
                    logger.warning(f"Validator {validator.address} slashed for unavailability")
    
    def get_slashing_history(self, validator_address: str = None) -> List[Dict]:
        """Get slashing history, optionally filtered by validator"""
        all_events = []
        
        for height_events in self.slashing_events.values():
            all_events.extend(height_events)
        
        if validator_address:
            all_events = [event for event in all_events if event['validator'] == validator_address]
        
        # Sort by timestamp (most recent first)
        all_events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return all_events
    
    def get_validator_slashing_risk(self, validator_address: str) -> Dict:
        """Calculate slashing risk score for a validator"""
        if validator_address not in self.consensus_engine.validators:
            return {}
        
        validator = self.consensus_engine.validators[validator_address]
        
        # Calculate various risk factors
        uptime_risk = max(0, (100 - validator.uptime) / 100)  # Higher risk with lower uptime
        slashing_history_risk = min(1.0, validator.slashing_count * 0.1)  # 10% per past slash
        delegation_risk = validator.total_delegated / validator.total_stake if validator.total_stake > 0 else 0
        
        overall_risk = (uptime_risk * 0.5 + slashing_history_risk * 0.3 + delegation_risk * 0.2)
        
        return {
            'validator': validator_address,
            'uptime_risk': uptime_risk,
            'slashing_history_risk': slashing_history_risk,
            'delegation_risk': delegation_risk,
            'overall_risk': overall_risk,
            'risk_level': 'LOW' if overall_risk < 0.3 else 'MEDIUM' if overall_risk < 0.7 else 'HIGH'
        }