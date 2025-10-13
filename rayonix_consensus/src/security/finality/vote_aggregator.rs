// consensus/security/finality/vote_aggregator.rs
use crate::types::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use bls::{Signature, PublicKey, AggregateSignature, SignatureSet};
use merlin::Transcript;
use statrs::{
    distribution::{Normal, Binomial, Poisson, Continuous},
    statistics::{Statistics, Distribution},
};
use nalgebra::{DVector, DMatrix, SVD, SymmetricEigen};

pub struct VoteAggregator {
    signature_aggregator: SignatureAggregator,
    vote_verifier: VoteVerifier,
    quorum_calculator: QuorumCalculator,
    equivocation_detector: EquivocationDetector,
    aggregation_optimizer: AggregationOptimizer,
}

impl VoteAggregator {
    pub async fn aggregate_votes_for_epoch(
        &self,
        votes: &[Vote],
        validators: &[ActiveValidator],
        epoch: Epoch,
        network_state: &NetworkState,
    ) -> Result<VoteAggregationResult, AggregationError> {
        // Phase 1: Verify and validate all votes
        let verified_votes = self.verify_and_validate_votes(votes, validators, epoch).await?;
        
        // Phase 2: Detect and handle equivocations
        let equivocation_free_votes = self.detect_and_handle_equivocations(&verified_votes, validators).await?;
        
        // Phase 3: Calculate voting power and quorum
        let quorum_analysis = self.calculate_quorum_analysis(&equivocation_free_votes, validators, network_state).await?;
        
        // Phase 4: Optimize signature aggregation
        let aggregation_strategy = self.optimize_aggregation_strategy(&equivocation_free_votes, validators).await?;
        
        // Phase 5: Perform signature aggregation
        let aggregate_signature = self.perform_signature_aggregation(&equivocation_free_votes, &aggregation_strategy).await?;
        
        // Phase 6: Generate aggregation proof
        let aggregation_proof = self.generate_aggregation_proof(
            &equivocation_free_votes, 
            &aggregate_signature, 
            epoch
        ).await?;

        Ok(VoteAggregationResult {
            aggregated_votes: equivocation_free_votes,
            aggregate_signature,
            aggregation_proof,
            quorum_analysis,
            aggregation_strategy,
            epoch,
            aggregation_quality: self.calculate_aggregation_quality(&equivocation_free_votes, &aggregate_signature).await?,
        })
    }

    async fn verify_and_validate_votes(
        &self,
        votes: &[Vote],
        validators: &[ActiveValidator],
        epoch: Epoch,
    ) -> Result<Vec<VerifiedVote>, AggregationError> {
        let verified_votes: Vec<VerifiedVote> = votes
            .par_iter()
            .filter_map(|vote| {
                match self.verify_single_vote(vote, validators, epoch).await {
                    Ok(verified_vote) => Some(verified_vote),
                    Err(_) => None, // Skip invalid votes
                }
            })
            .collect();

        if verified_votes.is_empty() {
            return Err(AggregationError::NoValidVotes);
        }

        Ok(verified_votes)
    }

    async fn verify_single_vote(
        &self,
        vote: &Vote,
        validators: &[ActiveValidator],
        epoch: Epoch,
    ) -> Result<VerifiedVote, AggregationError> {
        // Find validator and their public key
        let validator = validators.iter()
            .find(|v| v.identity.id == vote.voter_id)
            .ok_or(AggregationError::ValidatorNotFound(vote.voter_id))?;

        // Verify vote is for correct epoch
        if vote.target_epoch != epoch {
            return Err(AggregationError::WrongEpoch(vote.target_epoch, epoch));
        }

        // Verify signature
        let message = self.construct_vote_message(vote).await?;
        let signature_valid = self.vote_verifier.verify_signature(
            &vote.signature,
            &message,
            &validator.identity.consensus_key,
        ).await?;

        if !signature_valid {
            return Err(AggregationError::InvalidSignature(vote.voter_id));
        }

        // Verify voting power
        let voting_power = self.calculate_voting_power(validator, vote.target_epoch).await?;

        Ok(VerifiedVote {
            vote: vote.clone(),
            voter_public_key: validator.identity.consensus_key,
            voting_power,
            verification_timestamp: self.get_current_timestamp().await,
            verification_confidence: self.calculate_verification_confidence(validator).await?,
        })
    }

    async fn detect_and_handle_equivocations(
        &self,
        verified_votes: &[VerifiedVote],
        validators: &[ActiveValidator],
    ) -> Result<Vec<VerifiedVote>, AggregationError> {
        let mut vote_groups: HashMap<ValidatorId, Vec<&VerifiedVote>> = HashMap::new();
        
        // Group votes by validator
        for vote in verified_votes {
            vote_groups.entry(vote.vote.voter_id)
                .or_insert_with(Vec::new)
                .push(vote);
        }

        let mut equivocation_free_votes = Vec::new();
        let mut equivocation_detections = Vec::new();

        for (validator_id, votes) in vote_groups {
            if votes.len() == 1 {
                // Single vote - no equivocation
                equivocation_free_votes.push(votes[0].clone());
            } else {
                // Multiple votes - detect equivocation
                let equivocation_result = self.detect_equivocation_for_validator(&votes, validator_id).await?;
                
                if let Some(slashable_equivocation) = equivocation_result.slashable_equivocation {
                    equivocation_detections.push(slashable_equivocation);
                }
                
                // Include only the first valid vote (or apply other selection logic)
                if let Some(selected_vote) = self.select_canonical_vote(&votes).await? {
                    equivocation_free_votes.push(selected_vote);
                }
            }
        }

        // Report equivocations for slashing
        if !equivocation_detections.is_empty() {
            self.report_equivocations(&equivocation_detections).await?;
        }

        Ok(equivocation_free_votes)
    }

    async fn detect_equivocation_for_validator(
        &self,
        votes: &[&VerifiedVote],
        validator_id: ValidatorId,
    ) -> Result<EquivocationDetection, AggregationError> {
        let mut conflicting_votes = Vec::new();

        // Check all pairs of votes for conflicts
        for i in 0..votes.len() {
            for j in i + 1..votes.len() {
                if self.are_votes_conflicting(votes[i], votes[j]).await? {
                    conflicting_votes.push((votes[i].clone(), votes[j].clone()));
                }
            }
        }

        if conflicting_votes.is_empty() {
            return Ok(EquivocationDetection::no_equivocation());
        }

        // Determine if equivocation is slashable
        let slashable_equivocation = if conflicting_votes.len() >= 2 {
            Some(SlashableEquivocation {
                validator_id,
                conflicting_votes: conflicting_votes.iter()
                    .flat_map(|(v1, v2)| vec![v1.vote.clone(), v2.vote.clone()])
                    .collect(),
                detection_timestamp: self.get_current_timestamp().await,
                confidence: self.calculate_equivocation_confidence(&conflicting_votes).await?,
            })
        } else {
            None
        };

        Ok(EquivocationDetection {
            validator_id,
            conflicting_votes,
            slashable_equivocation,
            equivocation_type: self.determine_equivocation_type(&conflicting_votes).await?,
        })
    }

    async fn calculate_quorum_analysis(
        &self,
        votes: &[VerifiedVote],
        validators: &[ActiveValidator],
        network_state: &NetworkState,
    ) -> Result<QuorumAnalysis, AggregationError> {
        let total_voting_power: u128 = validators.iter()
            .map(|v| self.calculate_voting_power(v, network_state.current_epoch).await?)
            .sum();

        let accumulated_voting_power: u128 = votes.iter()
            .map(|v| v.voting_power)
            .sum();

        let voting_ratio = accumulated_voting_power as f64 / total_voting_power as f64;

        // Calculate Byzantine fault tolerance thresholds
        let safety_threshold = self.calculate_safety_threshold(network_state).await?;
        let liveness_threshold = self.calculate_liveness_threshold(network_state).await?;

        let quorum_achieved = voting_ratio >= safety_threshold;
        let supermajority_achieved = voting_ratio >= (2.0 / 3.0);

        // Calculate statistical confidence
        let statistical_confidence = self.calculate_quorum_confidence(
            votes.len(),
            validators.len(),
            voting_ratio,
            network_state
        ).await?;

        Ok(QuorumAnalysis {
            total_voting_power,
            accumulated_voting_power,
            voting_ratio,
            safety_threshold,
            liveness_threshold,
            quorum_achieved,
            supermajority_achieved,
            statistical_confidence,
            voter_distribution: self.analyze_voter_distribution(votes, validators).await?,
        })
    }

    async fn calculate_quorum_confidence(
        &self,
        voter_count: usize,
        total_validators: usize,
        voting_ratio: f64,
        network_state: &NetworkState,
    ) -> Result<f64, AggregationError> {
        // Use binomial distribution to calculate confidence in quorum
        let n = total_validators;
        let k = voter_count;
        let p = voting_ratio;

        if n == 0 {
            return Ok(0.0);
        }

        // Calculate probability that true support is above threshold
        let threshold = network_state.safety_threshold;
        
        // Bayesian approach with beta prior
        let alpha_prior = 2.0; // Weak prior
        let beta_prior = 2.0;
        
        let alpha_posterior = alpha_prior + k as f64;
        let beta_posterior = beta_prior + (n - k) as f64;
        
        // Calculate probability that true p > threshold using beta distribution
        let confidence = self.calculate_beta_tail_probability(
            alpha_posterior,
            beta_posterior,
            threshold
        ).await?;

        Ok(confidence.max(0.0).min(1.0))
    }

    async fn optimize_aggregation_strategy(
        &self,
        votes: &[VerifiedVote],
        validators: &[ActiveValidator],
    ) -> Result<AggregationStrategy, AggregationError> {
        let vote_count = votes.len();
        
        // Strategy 1: Simple aggregation (all signatures)
        let simple_strategy = AggregationStrategy::Simple {
            signature_count: vote_count,
            expected_gas: self.estimate_gas_cost(vote_count).await?,
        };

        // Strategy 2: Threshold signature scheme
        let threshold_strategy = if vote_count >= 100 {
            Some(self.calculate_threshold_strategy(votes, validators).await?)
        } else {
            None
        };

        // Strategy 3: Committee-based aggregation
        let committee_strategy = self.calculate_committee_strategy(votes, validators).await?;

        // Select optimal strategy based on cost and security
        let optimal_strategy = self.select_optimal_strategy(
            &simple_strategy,
            threshold_strategy.as_ref(),
            &committee_strategy,
            votes
        ).await?;

        Ok(optimal_strategy)
    }

    async fn perform_signature_aggregation(
        &self,
        votes: &[VerifiedVote],
        strategy: &AggregationStrategy,
    ) -> Result<AggregateSignature, AggregationError> {
        match strategy {
            AggregationStrategy::Simple { .. } => {
                self.aggregate_all_signatures(votes).await
            }
            AggregationStrategy::Threshold { threshold, .. } => {
                self.aggregate_threshold_signatures(votes, *threshold).await
            }
            AggregationStrategy::Committee { committees, .. } => {
                self.aggregate_committee_signatures(votes, committees).await
            }
        }
    }

    async fn aggregate_all_signatures(
        &self,
        votes: &[VerifiedVote],
    ) -> Result<AggregateSignature, AggregationError> {
        let signatures: Vec<Signature> = votes.iter()
            .map(|v| v.vote.signature.clone())
            .collect();

        AggregateSignature::aggregate(&signatures)
            .map_err(|e| AggregationError::SignatureAggregationFailed(e.to_string()))
    }

    async fn aggregate_threshold_signatures(
        &self,
        votes: &[VerifiedVote],
        threshold: usize,
    ) -> Result<AggregateSignature, AggregationError> {
        // Select top votes by voting power
        let mut sorted_votes = votes.to_vec();
        sorted_votes.sort_by(|a, b| b.voting_power.cmp(&a.voting_power));
        
        let selected_votes = &sorted_votes[..threshold.min(sorted_votes.len())];
        
        self.aggregate_all_signatures(selected_votes).await
    }

    async fn generate_aggregation_proof(
        &self,
        votes: &[VerifiedVote],
        aggregate_signature: &AggregateSignature,
        epoch: Epoch,
    ) -> Result<AggregationProof, AggregationError> {
        let mut transcript = Transcript::new(b"RAYONIX_VOTE_AGGREGATION_PROOF");
        
        // Commit to aggregation parameters
        transcript.append_message(b"epoch", &epoch.to_be_bytes());
        transcript.append_message(b"vote_count", &(votes.len() as u64).to_be_bytes());
        
        // Commit to voter set
        let voter_merkle_root = self.calculate_voter_merkle_root(votes).await?;
        transcript.append_message(b"voter_merkle_root", &voter_merkle_root.0);
        
        // Commit to aggregate signature
        let signature_bytes = aggregate_signature.serialize();
        transcript.append_message(b"aggregate_signature", &signature_bytes);
        
        // Generate proof challenge
        let mut challenge = [0u8; 32];
        transcript.challenge_bytes(b"aggregation_challenge", &mut challenge);
        
        Ok(AggregationProof {
            challenge,
            voter_merkle_root,
            aggregate_signature: aggregate_signature.clone(),
            vote_count: votes.len() as u32,
            proof_timestamp: self.get_current_timestamp().await,
        })
    }
}

pub struct SignatureAggregator {
    batch_verifier: BatchSignatureVerifier,
    aggregation_optimizer: AggregationOptimizer,
    gas_estimator: GasEstimator,
}

impl SignatureAggregator {
    pub async fn verify_signature_batch(
        &self,
        signatures: &[Signature],
        messages: &[Vec<u8>],
        public_keys: &[PublicKey],
    ) -> Result<BatchVerificationResult, AggregationError> {
        if signatures.len() != messages.len() || signatures.len() != public_keys.len() {
            return Err(AggregationError::BatchSizeMismatch);
        }

        // Use batch verification for efficiency
        let batch_valid = self.batch_verifier.verify_batch(signatures, messages, public_keys).await?;
        
        if batch_valid {
            return Ok(BatchVerificationResult {
                valid: true,
                invalid_indices: Vec::new(),
                verification_gas: self.gas_estimator.estimate_batch_verification_gas(signatures.len()).await?,
            });
        }

        // If batch verification fails, identify invalid signatures
        let invalid_indices = self.identify_invalid_signatures(signatures, messages, public_keys).await?;
        
        Ok(BatchVerificationResult {
            valid: false,
            invalid_indices,
            verification_gas: self.gas_estimator.estimate_individual_verification_gas(signatures.len()).await?,
        })
    }

    async fn identify_invalid_signatures(
        &self,
        signatures: &[Signature],
        messages: &[Vec<u8>],
        public_keys: &[PublicKey],
    ) -> Result<Vec<usize>, AggregationError> {
        let mut invalid_indices = Vec::new();
        
        for i in 0..signatures.len() {
            let valid = self.batch_verifier.verify_single(
                &signatures[i],
                &messages[i],
                &public_keys[i],
            ).await?;
            
            if !valid {
                invalid_indices.push(i);
            }
        }
        
        Ok(invalid_indices)
    }
}

pub struct QuorumCalculator {
    fault_tolerance_analyzer: FaultToleranceAnalyzer,
    statistical_analyzer: StatisticalAnalyzer,
    network_model: NetworkModel,
}

impl QuorumCalculator {
    pub async fn calculate_safety_threshold(
        &self,
        network_state: &NetworkState,
    ) -> Result<f64, AggregationError> {
        // Safety requires 2/3 honest in synchronous networks
        let base_threshold = 2.0 / 3.0;
        
        // Adjust for network synchrony
        let synchrony_adjustment = self.calculate_synchrony_adjustment(network_state).await?;
        
        // Adjust for message delay
        let delay_adjustment = self.calculate_delay_adjustment(network_state).await?;
        
        let adjusted_threshold = base_threshold * synchrony_adjustment * delay_adjustment;
        
        Ok(adjusted_threshold.max(0.5).min(0.9)) // Practical bounds
    }

    pub async fn calculate_liveness_threshold(
        &self,
        network_state: &NetworkState,
    ) -> Result<f64, AggregationError> {
        // Liveness requires 1/2 honest in partially synchronous networks
        let base_threshold = 1.0 / 2.0;
        
        // Adjust for network conditions
        let network_adjustment = self.calculate_network_adjustment(network_state).await?;
        
        let adjusted_threshold = base_threshold * network_adjustment;
        
        Ok(adjusted_threshold.max(0.3).min(0.7))
    }

    async fn calculate_synchrony_adjustment(
        &self,
        network_state: &NetworkState,
    ) -> Result<f64, AggregationError> {
        // Calculate network synchrony based on message propagation times
        let avg_propagation_time = network_state.average_propagation_time_ms;
        let max_propagation_time = network_state.max_propagation_time_ms;
        
        if max_propagation_time == 0.0 {
            return Ok(1.0);
        }
        
        let synchrony_ratio = avg_propagation_time / max_propagation_time;
        
        // More synchronous networks can use lower thresholds
        let adjustment = if synchrony_ratio < 0.3 {
            0.95 // Highly synchronous
        } else if synchrony_ratio < 0.7 {
            1.0 // Moderately synchronous
        } else {
            1.05 // Less synchronous
        };
        
        Ok(adjustment)
    }
}

pub struct EquivocationDetector {
    conflict_analyzer: ConflictAnalyzer,
    temporal_analyzer: TemporalAnalyzer,
    graph_analyzer: GraphAnalyzer,
}

impl EquivocationDetector {
    pub async fn are_votes_conflicting(
        &self,
        vote1: &VerifiedVote,
        vote2: &VerifiedVote,
    ) -> Result<bool, AggregationError> {
        if vote1.vote.voter_id != vote2.vote.voter_id {
            return Ok(false); // Different validators cannot equivocate
        }

        // Check for same target different hashes (classic equivocation)
        if vote1.vote.target_epoch == vote2.vote.target_epoch &&
           vote1.vote.target_hash != vote2.vote.target_hash {
            return Ok(true);
        }

        // Check for surround votes (more complex equivocation)
        if self.are_votes_surrounding(vote1, vote2).await? {
            return Ok(true);
        }

        // Check for temporal conflicts
        if self.have_temporal_conflict(vote1, vote2).await? {
            return Ok(true);
        }

        Ok(false)
    }

    async fn are_votes_surrounding(
        &self,
        vote1: &VerifiedVote,
        vote2: &VerifiedVote,
    ) -> Result<bool, AggregationError> {
        // Check if one vote surrounds another in the consensus graph
        // This requires access to the fork choice rule and consensus state
        
        let relationship = self.conflict_analyzer.analyze_vote_relationship(
            &vote1.vote,
            &vote2.vote,
        ).await?;

        Ok(relationship.is_surrounding())
    }

    async fn have_temporal_conflict(
        &self,
        vote1: &VerifiedVote,
        vote2: &VerifiedVote,
    ) -> Result<bool, AggregationError> {
        // Votes are temporally conflicting if they were cast in overlapping time windows
        // where only one should be possible
        
        let time_overlap = self.temporal_analyzer.calculate_time_overlap(
            vote1.vote.timestamp,
            vote2.vote.timestamp,
        ).await?;

        // If votes overlap significantly and conflict, it's equivocation
        Ok(time_overlap > 0.8 && self.are_votes_logically_conflicting(vote1, vote2).await?)
    }
}

// Advanced statistical analysis for quorum confidence
async fn calculate_beta_tail_probability(
    &self,
    alpha: f64,
    beta: f64,
    threshold: f64,
) -> Result<f64, AggregationError> {
    use statrs::distribution::Beta as BetaDist;
    
    let beta_dist = BetaDist::new(alpha, beta)
        .map_err(|e| AggregationError::StatisticalError(e.to_string()))?;
    
    // Probability that true parameter > threshold
    let tail_probability = 1.0 - beta_dist.cdf(threshold);
    
    Ok(tail_probability.max(0.0).min(1.0))
}