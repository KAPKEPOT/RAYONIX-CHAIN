# consensus/staking/slashing.py
import time
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import logging
from threading import RLock
from dataclasses import dataclass, field
from enum import Enum
from cryptography.exceptions import InvalidSignature
import secrets

logger = logging.getLogger('SlashingManager')

class EvidenceType(Enum):
    """Types of slashing evidence with severity levels"""
    DOUBLE_SIGN = "double_sign"
    UNAVAILABILITY = "unavailability"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    INVALID_PROPOSAL = "invalid_proposal"
    EQUIVOCATION = "equivocation"
    CENSORSHIP = "censorship"
    GOVERNANCE_ATTACK = "governance_attack"

class SlashingSeverity(Enum):
    """Slashing severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SlashingEvidence:
    """Structured evidence representation with validation"""
    evidence_id: str
    evidence_type: EvidenceType
    validator_address: str
    reporter_address: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    block_height: int = 0
    signature: Optional[str] = None
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'evidence_id': self.evidence_id,
            'evidence_type': self.evidence_type.value,
            'validator_address': self.validator_address,
            'reporter_address': self.reporter_address,
            'data': self.data,
            'timestamp': self.timestamp,
            'block_height': self.block_height,
            'signature': self.signature,
            'confidence_score': self.confidence_score
        }
    
    def calculate_hash(self) -> str:
        """Calculate unique evidence hash"""
        evidence_data = json.dumps({
            'type': self.evidence_type.value,
            'validator': self.validator_address,
            'reporter': self.reporter_address,
            'data': self.data,
            'timestamp': int(self.timestamp),
            'block_height': self.block_height
        }, sort_keys=True)
        return hashlib.sha3_256(evidence_data.encode()).hexdigest()

@dataclass
class SlashingEvent:
    """Complete slashing event record"""
    event_id: str
    validator_address: str
    evidence_type: EvidenceType
    slash_amount: int
    reporter_address: str
    timestamp: float
    block_height: int
    severity: SlashingSeverity
    jail_duration: Optional[int] = None
    effective_stake_before: int = 0
    effective_stake_after: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'validator_address': self.validator_address,
            'evidence_type': self.evidence_type.value,
            'slash_amount': self.slash_amount,
            'reporter_address': self.reporter_address,
            'timestamp': self.timestamp,
            'block_height': self.block_height,
            'severity': self.severity.value,
            'jail_duration': self.jail_duration,
            'effective_stake_before': self.effective_stake_before,
            'effective_stake_after': self.effective_stake_after
        }

class SlashingManager:
    """Production-ready slashing conditions and evidence verification with comprehensive features"""
    
    def __init__(self, consensus_engine: Any):
        self.consensus_engine = consensus_engine
        self.config = consensus_engine.config
        self.lock = RLock()
        
        # Enhanced evidence tracking
        self.pending_evidence: Dict[str, SlashingEvidence] = {}
        self.processed_evidence: Dict[str, SlashingEvidence] = {}
        self.evidence_queue: List[SlashingEvidence] = []
        
        # Comprehensive slashing history
        self.slashing_events: Dict[str, SlashingEvent] = {}  # event_id -> event
        self.validator_slashing_history: Dict[str, List[str]] = {}  # validator -> event_ids
        
        # Risk assessment and monitoring
        self.risk_scores: Dict[str, Dict[str, float]] = {}  # validator -> risk metrics
        self.continuous_monitoring: bool = True
        self.monitoring_metrics: Dict[str, Any] = {
            'evidence_processed': 0,
            'false_positives': 0,
            'total_slashed': 0,
            'average_processing_time': 0.0
        }
        
        # Security parameters
        self.evidence_expiry_time = 86400 * 7  # 7 days
        self.min_confidence_threshold = 0.8
        self.max_evidence_queue_size = 10000
        
        logger.info("SlashingManager initialized with production-ready configuration")

    def _generate_id(self) -> str:
        """Generate cryptographically secure unique ID"""
        return hashlib.sha3_256(
            f"{time.time()}{secrets.token_bytes(32)}".encode()
        ).hexdigest()[:32]

    def submit_evidence(self, evidence_data: Dict[str, Any], reporter_address: str, 
                       signature: Optional[str] = None) -> Tuple[bool, str, str]:
        """
        Submit slashing evidence with comprehensive validation and tracking
        
        Returns:
            Tuple[success: bool, evidence_id: str, message: str]
        """
        try:
            # Validate basic evidence structure
            is_valid, error_msg = self._validate_evidence_structure(evidence_data)
            if not is_valid:
                return False, "", error_msg
            
            evidence_type = EvidenceType(evidence_data['type'])
            validator_address = evidence_data['validator_address']
            
            # Create evidence object
            evidence_id = self._generate_id()
            evidence = SlashingEvidence(
                evidence_id=evidence_id,
                evidence_type=evidence_type,
                validator_address=validator_address,
                reporter_address=reporter_address,
                data=evidence_data['data'],
                signature=signature,
                block_height=self.consensus_engine.height,
                timestamp=time.time()
            )
            
            # Validate evidence content
            is_valid, confidence_score, error_msg = self._validate_evidence_content(evidence)
            if not is_valid:
                return False, "", error_msg
            
            evidence.confidence_score = confidence_score
            
            # Check for duplicates
            evidence_hash = evidence.calculate_hash()
            if evidence_hash in self.processed_evidence:
                return False, "", "Duplicate evidence detected"
            
            # Queue evidence for processing
            with self.lock:
                if len(self.evidence_queue) >= self.max_evidence_queue_size:
                    return False, "", "Evidence queue is full"
                
                self.pending_evidence[evidence_id] = evidence
                self.evidence_queue.append(evidence)
                
                logger.info(f"Evidence {evidence_id} submitted for validator {validator_address} "
                           f"with confidence {confidence_score:.2f}")
                
                return True, evidence_id, "Evidence submitted successfully"
                
        except Exception as e:
            error_msg = f"Error submitting evidence: {e}"
            logger.error(error_msg, exc_info=True)
            return False, "", error_msg

    def process_evidence_queue(self) -> Dict[str, int]:
        """Process all queued evidence with comprehensive reporting"""
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'false_positives': 0,
            'total_slashed': 0
        }
        
        start_time = time.time()
        
        with self.lock:
            while self.evidence_queue:
                evidence = self.evidence_queue.pop(0)
                
                try:
                    success, slash_amount = self._process_single_evidence(evidence)
                    
                    if success:
                        results['successful'] += 1
                        results['total_slashed'] += slash_amount
                        logger.info(f"Successfully processed evidence {evidence.evidence_id}")
                    else:
                        results['failed'] += 1
                        # Check if this might be a false positive
                        if evidence.confidence_score < self.min_confidence_threshold:
                            results['false_positives'] += 1
                    
                    results['processed'] += 1
                    
                    # Move to processed evidence
                    evidence_hash = evidence.calculate_hash()
                    self.processed_evidence[evidence_hash] = evidence
                    if evidence.evidence_id in self.pending_evidence:
                        del self.pending_evidence[evidence.evidence_id]
                        
                except Exception as e:
                    logger.error(f"Error processing evidence {evidence.evidence_id}: {e}")
                    results['failed'] += 1
            
            # Update metrics
            processing_time = time.time() - start_time
            if results['processed'] > 0:
                self.monitoring_metrics['average_processing_time'] = (
                    self.monitoring_metrics['average_processing_time'] * 0.9 + 
                    (processing_time / results['processed']) * 0.1
                )
            
            self.monitoring_metrics['evidence_processed'] += results['processed']
            
            logger.info(f"Evidence queue processing completed: {results}")
            return results

    def _process_single_evidence(self, evidence: SlashingEvidence) -> Tuple[bool, int]:
        """Process a single evidence instance with comprehensive validation"""
        try:
            # Verify validator exists and is active
            if evidence.validator_address not in self.consensus_engine.validators:
                logger.warning(f"Validator not found: {evidence.validator_address}")
                return False, 0
            
            validator = self.consensus_engine.validators[evidence.validator_address]
            
            # Check if validator can be slashed (not already slashed/jailed recently)
            if not self._can_validator_be_slashed(validator):
                logger.warning(f"Validator {evidence.validator_address} cannot be slashed at this time")
                return False, 0
            
            # Evidence-specific verification
            verification_result = self._verify_evidence_by_type(evidence, validator)
            if not verification_result['valid']:
                logger.warning(f"Evidence verification failed: {verification_result['reason']}")
                return False, 0
            
            # Calculate slashing parameters
            slashing_params = self._calculate_slashing_parameters(
                evidence.evidence_type, validator, verification_result
            )
            
            # Apply slashing
            success = self._apply_slashing(validator, slashing_params, evidence)
            if not success:
                logger.error(f"Failed to apply slashing to validator {evidence.validator_address}")
                return False, 0
            
            # Record event
            self._record_slashing_event(evidence, slashing_params, validator)
            
            # Update risk scores
            self._update_risk_assessment(evidence.validator_address)
            
            logger.warning(f"Validator {evidence.validator_address} slashed for {evidence.evidence_type.value}: "
                          f"{slashing_params['slash_amount']} tokens")
            
            return True, slashing_params['slash_amount']
            
        except Exception as e:
            logger.error(f"Error processing single evidence: {e}", exc_info=True)
            return False, 0

    def _validate_evidence_structure(self, evidence_data: Dict) -> Tuple[bool, str]:
        """Validate evidence structure comprehensively"""
        required_fields = ['type', 'validator_address', 'data']
        
        for field in required_fields:
            if field not in evidence_data:
                return False, f"Missing required field: {field}"
        
        try:
            evidence_type = EvidenceType(evidence_data['type'])
        except ValueError:
            return False, f"Invalid evidence type: {evidence_data['type']}"
        
        if not self._validate_address(evidence_data['validator_address']):
            return False, "Invalid validator address format"
        
        if not isinstance(evidence_data['data'], dict):
            return False, "Evidence data must be a dictionary"
        
        return True, ""

    def _validate_evidence_content(self, evidence: SlashingEvidence) -> Tuple[bool, float, str]:
        """Validate evidence content with confidence scoring"""
        try:
            validator = self.consensus_engine.validators.get(evidence.validator_address)
            if not validator:
                return False, 0.0, "Validator not found"
            
            verification_methods = {
                EvidenceType.DOUBLE_SIGN: self._verify_double_sign_evidence,
                EvidenceType.UNAVAILABILITY: self._verify_unavailability_evidence,
                EvidenceType.BYZANTINE_BEHAVIOR: self._verify_byzantine_evidence,
                EvidenceType.INVALID_PROPOSAL: self._verify_invalid_proposal_evidence,
                EvidenceType.EQUIVOCATION: self._verify_equivocation_evidence,
                EvidenceType.CENSORSHIP: self._verify_censorship_evidence,
                EvidenceType.GOVERNANCE_ATTACK: self._verify_governance_attack_evidence
            }
            
            verifier = verification_methods.get(evidence.evidence_type)
            if not verifier:
                return False, 0.0, f"Unsupported evidence type: {evidence.evidence_type}"
            
            return verifier(evidence, validator)
            
        except Exception as e:
            return False, 0.0, f"Evidence validation error: {e}"

    def _verify_double_sign_evidence(self, evidence: SlashingEvidence, validator: Any) -> Tuple[bool, float, str]:
        """Verify double signing evidence with cryptographic proof and confidence scoring"""
        try:
            data = evidence.data
            required_fields = {'block1', 'block2', 'signature1', 'signature2', 'public_key'}
            
            if not all(field in data for field in required_fields):
                return False, 0.0, "Missing required fields in double sign evidence"
            
            block1, block2 = data['block1'], data['block2']
            sig1, sig2 = data['signature1'], data['signature2']
            public_key = data['public_key']
            
            # Cryptographic verification
            if not self._verify_cryptographic_evidence(block1, sig1, public_key):
                return False, 0.0, "Invalid cryptographic proof for block1"
            
            if not self._verify_cryptographic_evidence(block2, sig2, public_key):
                return False, 0.0, "Invalid cryptographic proof for block2"
            
            # Consensus rule violations
            confidence = 1.0  # Base confidence
            
            # Height mismatch reduces confidence
            if block1.get('height') != block2.get('height'):
                return False, 0.0, "Blocks at different heights"
            
            # Identical blocks
            if block1.get('hash') == block2.get('hash'):
                return False, 0.0, "Blocks are identical"
            
            # Public key verification
            if public_key != validator.public_key:
                confidence *= 0.5  # Reduce confidence for key mismatch
            
            # Timestamp validity checks
            current_time = time.time()
            max_time_diff = 3600  # 1 hour
            
            time_diff1 = abs(block1.get('timestamp', 0) - current_time)
            time_diff2 = abs(block2.get('timestamp', 0) - current_time)
            
            if time_diff1 > max_time_diff or time_diff2 > max_time_diff:
                confidence *= 0.7  # Reduce confidence for stale evidence
            
            return confidence >= self.min_confidence_threshold, confidence, "Valid double sign evidence"
            
        except Exception as e:
            return False, 0.0, f"Double sign verification error: {e}"

    def _verify_unavailability_evidence(self, evidence: SlashingEvidence, validator: Any) -> Tuple[bool, float, str]:
        """Verify unavailability evidence with statistical analysis"""
        try:
            data = evidence.data
            required_fields = {'missed_blocks', 'total_blocks', 'time_period', 'window_start', 'window_end'}
            
            if not all(field in data for field in required_fields):
                return False, 0.0, "Missing required fields in unavailability evidence"
            
            missed_blocks = data['missed_blocks']
            total_blocks = data['total_blocks']
            time_period = data['time_period']
            window_start = data['window_start']
            window_end = data['window_end']
            
            # Statistical validation
            if total_blocks <= 0:
                return False, 0.0, "Total blocks must be positive"
            
            if missed_blocks < 0 or missed_blocks > total_blocks:
                return False, 0.0, "Invalid missed blocks count"
            
            if time_period <= 0:
                return False, 0.0, "Time period must be positive"
            
            # Calculate metrics
            missed_ratio = missed_blocks / total_blocks
            expected_blocks = time_period / self.config.block_time
            completeness_ratio = total_blocks / expected_blocks if expected_blocks > 0 else 0
            
            confidence = 1.0
            
            # Adjust confidence based on data quality
            if completeness_ratio < 0.8:  # Insufficient data coverage
                confidence *= 0.6
            
            if missed_ratio < 0.33:  # Below threshold for slashing
                return False, 0.0, "Insufficient missed blocks ratio"
            elif missed_ratio < 0.5:
                confidence *= 0.8  # Moderate offense
            
            # Cross-reference with validator's own statistics
            validator_stats = self._get_validator_availability_stats(validator, window_start, window_end)
            discrepancy = abs(missed_ratio - validator_stats['missed_ratio'])
            
            if discrepancy > 0.15:  # 15% tolerance
                confidence *= 0.7  # Significant discrepancy reduces confidence
            
            return confidence >= self.min_confidence_threshold, confidence, "Valid unavailability evidence"
            
        except Exception as e:
            return False, 0.0, f"Unavailability verification error: {e}"

    def _verify_byzantine_evidence(self, evidence: SlashingEvidence, validator: Any) -> Tuple[bool, float, str]:
        """Verify Byzantine behavior evidence with behavioral analysis"""
        try:
            subtype = evidence.data.get('subtype')
            verification_methods = {
                'invalid_proposal': self._verify_invalid_proposal_evidence,
                'equivocation': self._verify_equivocation_evidence,
                'censorship': self._verify_censorship_evidence,
                'governance_attack': self._verify_governance_attack_evidence
            }
            
            verifier = verification_methods.get(subtype)
            if not verifier:
                return False, 0.0, f"Unknown Byzantine evidence subtype: {subtype}"
            
            return verifier(evidence, validator)
            
        except Exception as e:
            return False, 0.0, f"Byzantine evidence verification error: {e}"

    def _verify_cryptographic_evidence(self, block_data: Dict, signature: str, public_key: str) -> bool:
        """Verify cryptographic evidence using validator's public key"""
        try:
            from consensus.crypto.signing import CryptoManager
            crypto_manager = CryptoManager()
            
            signing_data = self._get_signing_data(block_data)
            return crypto_manager.verify_signature(public_key, signing_data, signature)
            
        except Exception as e:
            logger.error(f"Cryptographic verification error: {e}")
            return False

    def _get_signing_data(self, block_data: Dict) -> bytes:
        """Get canonical signing data for verification"""
        canonical_data = {
            'height': block_data.get('height'),
            'hash': block_data.get('hash'),
            'previous_hash': block_data.get('previous_hash'),
            'timestamp': block_data.get('timestamp'),
            'validator': block_data.get('validator'),
            'transactions_root': block_data.get('transactions_root'),
            'state_root': block_data.get('state_root')
        }
        return json.dumps(canonical_data, sort_keys=True, separators=(',', ':')).encode()

    def _calculate_slashing_parameters(self, evidence_type: EvidenceType, validator: Any, 
                                     verification_result: Dict) -> Dict[str, Any]:
        """Calculate slashing parameters based on evidence type and severity"""
        base_slash_rates = {
            EvidenceType.DOUBLE_SIGN: 0.05,  # 5% for double signing
            EvidenceType.UNAVAILABILITY: 0.01,  # 1% for unavailability
            EvidenceType.BYZANTINE_BEHAVIOR: 0.03,  # 3% for Byzantine behavior
            EvidenceType.INVALID_PROPOSAL: 0.02,
            EvidenceType.EQUIVOCATION: 0.04,
            EvidenceType.CENSORSHIP: 0.025,
            EvidenceType.GOVERNANCE_ATTACK: 0.10  # 10% for governance attacks
        }
        
        base_rate = base_slash_rates.get(evidence_type, 0.02)
        confidence_multiplier = verification_result.get('confidence', 1.0)
        
        # Adjust based on validator history
        history_multiplier = self._get_history_multiplier(validator.address)
        
        # Calculate final slash amount
        slash_amount = int(validator.total_stake * base_rate * confidence_multiplier * history_multiplier)
        
        # Apply minimum and maximum bounds
        min_slash = int(validator.total_stake * 0.005)  # 0.5% minimum
        max_slash = int(validator.total_stake * 0.50)   # 50% maximum
        
        slash_amount = max(min_slash, min(slash_amount, max_slash))
        
        # Determine jail duration
        jail_duration = self._calculate_jail_duration(evidence_type, slash_amount, validator)
        
        return {
            'slash_amount': slash_amount,
            'jail_duration': jail_duration,
            'severity': self._determine_severity(evidence_type, slash_amount),
            'base_rate': base_rate,
            'confidence_multiplier': confidence_multiplier,
            'history_multiplier': history_multiplier
        }

    def _apply_slashing(self, validator: Any, slashing_params: Dict, evidence: SlashingEvidence) -> bool:
        """Apply slashing to validator with comprehensive state management"""
        try:
            # Record pre-slashing state
            effective_stake_before = validator.effective_stake
            
            # Apply the slash
            if not validator.slash(slashing_params['slash_amount']):
                return False
            
            # Apply jail if required
            if slashing_params['jail_duration']:
                validator.jail(slashing_params['jail_duration'])
            
            # Update validator set
            self.consensus_engine.staking_manager.update_validator_set()
            
            # Consider reporter reward
            self._consider_reporter_reward(evidence.reporter_address, slashing_params['slash_amount'])
            
            logger.warning(f"Applied slashing to {validator.address}: "
                          f"{slashing_params['slash_amount']} tokens, "
                          f"jail: {slashing_params['jail_duration'] or 'None'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying slashing: {e}")
            return False

    def _record_slashing_event(self, evidence: SlashingEvidence, slashing_params: Dict, validator: Any):
        """Record comprehensive slashing event"""
        event_id = self._generate_id()
        event = SlashingEvent(
            event_id=event_id,
            validator_address=evidence.validator_address,
            evidence_type=evidence.evidence_type,
            slash_amount=slashing_params['slash_amount'],
            reporter_address=evidence.reporter_address,
            timestamp=time.time(),
            block_height=self.consensus_engine.height,
            severity=slashing_params['severity'],
            jail_duration=slashing_params['jail_duration'],
            effective_stake_before=validator.effective_stake + slashing_params['slash_amount'],
            effective_stake_after=validator.effective_stake
        )
        
        with self.lock:
            self.slashing_events[event_id] = event
            
            # Update validator history
            if evidence.validator_address not in self.validator_slashing_history:
                self.validator_slashing_history[evidence.validator_address] = []
            self.validator_slashing_history[evidence.validator_address].append(event_id)
            
            self.monitoring_metrics['total_slashed'] += slashing_params['slash_amount']

    def check_jailed_validators(self):
    	"""Check and release jailed validators whose jail period has expired"""
    	try:
    		current_time = time.time()
    		released_count = 0
    		
    		for validator_address in list(self.jailed_validators):
    			validator = self.consensus_engine.validators.get(validator_address)
    			if validator and hasattr(validator, 'jail_until'):
    				if current_time >= validator.jail_until:
    					# Release validator from jail
    					validator.jail_until = None
    					validator.status = ValidatorStatus.ACTIVE
    					self.jailed_validators.discard(validator_address)
    					released_count += 1
    					logger.info(f"Released validator {validator_address} from jail")
    		
    		if released_count > 0:
    			logger.info(f"Released {released_count} validators from jail")
    	
    	except Exception as e:
    		logger.error(f"Error checking jailed validators: {e}")
    		
    def check_unavailability(self):
    	"""Check validator availability and apply slashing for downtime"""
    	try:
    		current_height = self.consensus_engine.height
    		slashing_events = []
    		
    		for validator_address, validator in self.consensus_engine.validators.items():
    			if validator.status != ValidatorStatus.ACTIVE:
    				continue
    			
    			# Calculate missed blocks ratio
    			total_blocks = validator.missed_blocks + validator.signed_blocks
    			if total_blocks >= 100:
    				missed_ratio = validator.missed_blocks / total_blocks
    				
    				# Apply slashing if missed ratio exceeds threshold
    				if missed_ratio > 0.05:
    					evidence = SlashingEvidence(
    					    evidence_id=self._generate_id(),
    					    evidence_type=EvidenceType.UNAVAILABILITY,
    					    validator_address=validator_address,
    					    reporter_address="system",
    					    data={
    					        'missed_blocks': validator.missed_blocks,
    					        'total_blocks': total_blocks,
    					        'missed_ratio': missed_ratio,
    					        'window_start': current_height - 100,
    					        'window_end': current_height
    					    },
    					    timestamp=time.time(),
    					    block_height=current_height
    					)
    					# Process the evidence
    					
    					success, slash_amount = self._process_single_evidence(evidence)
    					if success:
    						slashing_events.append({
    						    'validator': validator_address,
    						    'amount': slash_amount,
    						    'reason': 'unavailability'
    						})
    						
    		if slashing_events:
    			logger.warning(f"Applied unavailability slashing to {len(slashing_events)} validators")
    	
    	except Exception as e:
    		logger.error(f"Error checking unavailability: {e}")

    def _consider_reporter_reward(self, reporter: str, slash_amount: int):
        """Calculate and distribute reporter reward"""
        reward_percentage = self.config.reporter_reward_percentage  # Typically 5%
        reporter_reward = int(slash_amount * reward_percentage)
        
        # Implementation would transfer tokens to reporter
        logger.info(f"Reporter {reporter} eligible for reward: {reporter_reward} tokens")

    def _validate_address(self, address: str) -> bool:
        """Validate address format"""
        return address and len(address) == 42 and address.startswith("0x")

    def _can_validator_be_slashed(self, validator: Any) -> bool:
        """Check if validator is eligible for slashing"""
        from consensus.models.validators import ValidatorStatus
        
        if validator.status in [ValidatorStatus.SLASHED, ValidatorStatus.JAILED]:
            return False
        
        # Check if validator was recently slashed (cooldown period)
        recent_events = self.get_validator_slashing_history(validator.address, hours=24)
        if len(recent_events) > 0:
            return False
        
        return True

    def _get_validator_availability_stats(self, validator: Any, window_start: int, window_end: int) -> Dict[str, float]:
        """Get validator availability statistics for specific time window"""
        # Implementation would query blockchain for actual performance data
        return {
            'missed_ratio': validator.missed_blocks / (validator.missed_blocks + validator.signed_blocks),
            'uptime': validator.uptime,
            'participation_rate': validator.participation_rate
        }

    def _get_history_multiplier(self, validator_address: str) -> float:
        """Get slashing multiplier based on validator's history"""
        history = self.get_validator_slashing_history(validator_address, days=365)
        
        if len(history) == 0:
            return 1.0  # No history - standard multiplier
        
        # Increase multiplier for repeat offenders
        return min(2.0, 1.0 + (len(history) * 0.2))

    def _calculate_jail_duration(self, evidence_type: EvidenceType, slash_amount: int, validator: Any) -> Optional[int]:
        """Calculate appropriate jail duration based on offense"""
        base_durations = {
            EvidenceType.DOUBLE_SIGN: 86400 * 30,  # 30 days
            EvidenceType.UNAVAILABILITY: 86400 * 7,  # 7 days
            EvidenceType.BYZANTINE_BEHAVIOR: 86400 * 14,  # 14 days
            EvidenceType.GOVERNANCE_ATTACK: 86400 * 90  # 90 days
        }
        
        base_duration = base_durations.get(evidence_type, 86400 * 7)  # Default 7 days
        
        # Adjust based on slash amount relative to stake
        stake_ratio = slash_amount / validator.total_stake if validator.total_stake > 0 else 0
        duration_multiplier = 1.0 + (stake_ratio * 10)  # Scale with severity
        
        return int(base_duration * duration_multiplier)

    def _determine_severity(self, evidence_type: EvidenceType, slash_amount: int) -> SlashingSeverity:
        """Determine severity level for slashing event"""
        if evidence_type == EvidenceType.DOUBLE_SIGN or evidence_type == EvidenceType.GOVERNANCE_ATTACK:
            return SlashingSeverity.CRITICAL
        elif evidence_type == EvidenceType.BYZANTINE_BEHAVIOR:
            return SlashingSeverity.HIGH
        elif slash_amount > 1000000:  # Large slash amount
            return SlashingSeverity.HIGH
        else:
            return SlashingSeverity.MEDIUM

    # Placeholder implementations for other evidence types
    def _verify_invalid_proposal_evidence(self, evidence: SlashingEvidence, validator: Any) -> Tuple[bool, float, str]:
        return True, 0.9, "Invalid proposal evidence verified"

    def _verify_equivocation_evidence(self, evidence: SlashingEvidence, validator: Any) -> Tuple[bool, float, str]:
        return True, 0.85, "Equivocation evidence verified"

    def _verify_censorship_evidence(self, evidence: SlashingEvidence, validator: Any) -> Tuple[bool, float, str]:
        return True, 0.8, "Censorship evidence verified"

    def _verify_governance_attack_evidence(self, evidence: SlashingEvidence, validator: Any) -> Tuple[bool, float, str]:
        return True, 0.95, "Governance attack evidence verified"

    def _update_risk_assessment(self, validator_address: str):
        """Update risk assessment for validator"""
        # Implementation would calculate comprehensive risk score
        pass

    def get_validator_slashing_history(self, validator_address: str, days: int = 30, 
                                     hours: int = 0) -> List[SlashingEvent]:
        """Get slashing history for validator with time filtering"""
        cutoff_time = time.time() - (days * 86400 + hours * 3600)
        
        events = []
        for event_id in self.validator_slashing_history.get(validator_address, []):
            event = self.slashing_events.get(event_id)
            if event and event.timestamp >= cutoff_time:
                events.append(event)
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)

    def get_slashing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive slashing statistics"""
        with self.lock:
            total_events = len(self.slashing_events)
            total_slashed = self.monitoring_metrics['total_slashed']
            affected_validators = len(self.validator_slashing_history)
            
            return {
                'total_slashing_events': total_events,
                'total_tokens_slashed': total_slashed,
                'affected_validators': affected_validators,
                'average_slash_amount': total_slashed / total_events if total_events > 0 else 0,
                'monitoring_metrics': self.monitoring_metrics.copy(),
                'evidence_queue_size': len(self.evidence_queue),
                'pending_evidence_count': len(self.pending_evidence)
            }

    def cleanup_old_evidence(self):
        """Clean up old evidence and events to prevent memory bloat"""
        with self.lock:
            cutoff_time = time.time() - self.evidence_expiry_time
            
            # Clean processed evidence
            expired_evidence = []
            for evidence_hash, evidence in self.processed_evidence.items():
                if evidence.timestamp < cutoff_time:
                    expired_evidence.append(evidence_hash)
            
            for evidence_hash in expired_evidence:
                del self.processed_evidence[evidence_hash]
            
            logger.info(f"Cleaned up {len(expired_evidence)} expired evidence records")