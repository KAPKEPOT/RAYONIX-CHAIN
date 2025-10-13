// consensus/security/slashing/detector.rs
use crate::types::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use bls::AggregateSignature;

pub struct SlashingDetector {
    equivocation_tracker: Arc<RwLock<BTreeMap<ValidatorId, BTreeMap<Slot, BlockHeader>>>>,
    surround_vote_tracker: Arc<RwLock<BTreeMap<ValidatorId, Vec<AttestationData>>>>,
    double_vote_tracker: Arc<RwLock<BTreeMap<ValidatorId, BTreeMap<Epoch, Attestation>>>>,
    slashing_queue: Arc<Mutex<VecDeque<SlashingOffense>>>,
    whistleblower_rewards: Arc<RwLock<BTreeMap<ValidatorId, u128>>>,
    evidence_verifier: EvidenceVerifier,
    penalty_escalation: PenaltyEscalationModel,
}

impl SlashingDetector {
    pub async fn check_block_proposal(
        &self,
        block: &Block,
        proposer: ValidatorId,
        current_slot: Slot,
    ) -> Result<SlashingCheckResult, SlashingError> {
        // Phase 1: Check for equivocation
        let equivocation_check = self.check_equivocation(proposer, &block.header, current_slot).await?;
        
        // Phase 2: Verify block signature
        let signature_check = self.verify_block_signature(block, proposer).await?;
        
        // Phase 3: Check for proposer boost manipulation
        let boost_manipulation_check = self.check_proposer_boost_manipulation(block, proposer).await?;
        
        // Phase 4: Verify block transactions ordering
        let tx_ordering_check = self.check_transaction_ordering_manipulation(block).await?;
        
        // Phase 5: Check for timestamp manipulation
        let timestamp_check = self.check_timestamp_manipulation(block, current_slot).await?;
        
        let offenses = vec![equivocation_check, boost_manipulation_check, tx_ordering_check, timestamp_check]
            .into_iter()
            .filter_map(|result| result.offense)
            .collect::<Vec<SlashingOffense>>();
        
        Ok(SlashingCheckResult {
            is_slashable: !offenses.is_empty(),
            offenses,
            severity: self.calculate_offense_severity(&offenses).await,
            required_action: self.determine_required_action(&offenses).await,
        })
    }
    
    pub async fn check_attestation(
        &self,
        attestation: &Attestation,
        attester: ValidatorId,
        current_epoch: Epoch,
    ) -> Result<SlashingCheckResult, SlashingError> {
        // Phase 1: Check for double voting
        let double_vote_check = self.check_double_voting(attester, attestation, current_epoch).await?;
        
        // Phase 2: Check for surround voting
        let surround_vote_check = self.check_surround_voting(attester, attestation).await?;
        
        // Phase 3: Check for ghost vote manipulation
        let ghost_vote_check = self.check_ghost_vote_manipulation(attestation, attester).await?;
        
        // Phase 4: Verify attestation signature
        let signature_check = self.verify_attestation_signature(attestation, attester).await?;
        
        let offenses = vec![double_vote_check, surround_vote_check, ghost_vote_check]
            .into_iter()
            .filter_map(|result| result.offense)
            .collect::<Vec<SlashingOffense>>();
        
        Ok(SlashingCheckResult {
            is_slashable: !offenses.is_empty(),
            offenses,
            severity: self.calculate_offense_severity(&offenses).await,
            required_action: self.determine_required_action(&offenses).await,
        })
    }
    
    async fn check_equivocation(
        &self,
        proposer: ValidatorId,
        header: &BlockHeader,
        current_slot: Slot,
    ) -> Result<SlashingCheckResult, SlashingError> {
        let mut tracker = self.equivocation_tracker.write().await;
        
        let proposer_blocks = tracker.entry(proposer).or_insert_with(BTreeMap::new);
        
        // Check if proposer already signed a block for this slot
        if let Some(existing_header) = proposer_blocks.get(&current_slot) {
            if existing_header != header {
                // Equivocation detected - different blocks for same slot
                let offense = SlashingOffense {
                    validator_id: proposer,
                    offense_type: OffenseType::Equivocation,
                    severity: OffenseSeverity::Critical,
                    evidence: SlashingEvidence::BlockEquivocation {
                        slot: current_slot,
                        first_block: existing_header.clone(),
                        second_block: header.clone(),
                    },
                    detection_epoch: self.get_current_epoch().await,
                    reporter: None, // Auto-detected by system
                    confidence_score: 1.0,
                };
                
                return Ok(SlashingCheckResult {
                    is_slashable: true,
                    offenses: vec![offense],
                    severity: OffenseSeverity::Critical,
                    required_action: RequiredAction::ImmediateJail,
                });
            }
        }
        
        // Store the block header for this slot
        proposer_blocks.insert(current_slot, header.clone());
        
        // Clean up old entries to prevent memory growth
        self.cleanup_old_equivocation_data(&mut tracker, current_slot).await;
        
        Ok(SlashingCheckResult::no_offense())
    }
    
    async fn check_double_voting(
        &self,
        attester: ValidatorId,
        new_attestation: &Attestation,
        current_epoch: Epoch,
    ) -> Result<SlashingCheckResult, SlashingError> {
        let mut tracker = self.double_vote_tracker.write().await;
        
        let attester_votes = tracker.entry(attester).or_insert_with(BTreeMap::new);
        let epoch_votes = attester_votes.entry(current_epoch).or_insert_with(Vec::new);
        
        // Check for conflicting attestations within the same epoch
        for existing_attestation in epoch_votes.iter() {
            if self.are_attestations_conflicting(existing_attestation, new_attestation).await {
                // Double voting detected
                let offense = SlashingOffense {
                    validator_id: attester,
                    offense_type: OffenseType::DoubleVoting,
                    severity: OffenseSeverity::Critical,
                    evidence: SlashingEvidence::AttestationConflict {
                        epoch: current_epoch,
                        first_attestation: existing_attestation.clone(),
                        second_attestation: new_attestation.clone(),
                    },
                    detection_epoch: self.get_current_epoch().await,
                    reporter: None,
                    confidence_score: 1.0,
                };
                
                return Ok(SlashingCheckResult {
                    is_slashable: true,
                    offenses: vec![offense],
                    severity: OffenseSeverity::Critical,
                    required_action: RequiredAction::ImmediateJail,
                });
            }
        }
        
        // Store the new attestation
        epoch_votes.push(new_attestation.clone());
        
        Ok(SlashingCheckResult::no_offense())
    }
    
    async fn check_surround_voting(
        &self,
        attester: ValidatorId,
        new_attestation: &Attestation,
    ) -> Result<SlashingCheckResult, SlashingError> {
        let mut tracker = self.surround_vote_tracker.write().await;
        
        let attester_history = tracker.entry(attester).or_insert_with(Vec::new);
        
        for existing_attestation in attester_history.iter() {
            let surround_relationship = self.analyze_attestation_relationship(existing_attestation, &new_attestation.data).await;
            
            if surround_relationship.is_surrounding() {
                let offense = SlashingOffense {
                    validator_id: attester,
                    offense_type: OffenseType::SurroundVoting,
                    severity: OffenseSeverity::High,
                    evidence: SlashingEvidence::SurroundVote {
                        existing_attestation: existing_attestation.clone(),
                        new_attestation: new_attestation.data.clone(),
                        relationship: surround_relationship,
                    },
                    detection_epoch: self.get_current_epoch().await,
                    reporter: None,
                    confidence_score: 0.95, // Slightly lower confidence due to complexity
                };
                
                return Ok(SlashingCheckResult {
                    is_slashable: true,
                    offenses: vec![offense],
                    severity: OffenseSeverity::High,
                    required_action: RequiredAction::InvestigationRequired,
                });
            }
        }
        
        // Store the new attestation data
        attester_history.push(new_attestation.data.clone());
        
        // Clean up old attestation data
        self.cleanup_old_attestation_data(&mut tracker).await;
        
        Ok(SlashingCheckResult::no_offense())
    }
    
    pub async fn process_whistleblower_report(
        &self,
        report: WhistleblowerReport,
        reporter: ValidatorId,
    ) -> Result<SlashingProcessingResult, SlashingError> {
        // Phase 1: Verify reporter credibility
        let reporter_credibility = self.assess_reporter_credibility(reporter).await?;
        
        // Phase 2: Validate evidence
        let evidence_validation = self.validate_whistleblower_evidence(&report.evidence).await?;
        
        // Phase 3: Corroborate with system data
        let corroboration = self.corroborate_evidence(&report.evidence).await?;
        
        // Phase 4: Calculate confidence score
        let confidence_score = self.calculate_evidence_confidence(
            &report.evidence,
            reporter_credibility,
            corroboration,
        ).await;
        
        if confidence_score < self.whistleblower_config.min_confidence_threshold {
            return Err(SlashingError::InsufficientEvidenceConfidence(confidence_score));
        }
        
        // Phase 5: Create slashing offense
        let offense = SlashingOffense {
            validator_id: report.accused_validator,
            offense_type: report.offense_type,
            severity: self.determine_offense_severity(&report.evidence).await,
            evidence: report.evidence,
            detection_epoch: self.get_current_epoch().await,
            reporter: Some(reporter),
            confidence_score,
        };
        
        // Phase 6: Queue for penalty application
        self.queue_slashing_offense(offense.clone()).await?;
        
        // Phase 7: Calculate whistleblower reward
        let reward = self.calculate_whistleblower_reward(&offense, reporter).await?;
        
        Ok(SlashingProcessingResult {
            offense,
            reward_amount: reward,
            status: SlashingStatus::QueuedForProcessing,
            estimated_penalty: self.estimate_penalty_amount(&offense).await,
        })
    }
}