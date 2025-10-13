// consensus/security/finality/fork_choice.rs
use crate::types::*;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;

pub struct ForkChoiceRule {
    block_dag: BlockDag,
    justification_graph: JustificationGraph,
    weight_calculator: ForkChoiceWeightCalculator,
    lmd_ghost: LMDGhost,
    ffg_voting: FFGVoting,
    reorg_protector: ReorgProtector,
    chain_selection: ChainSelectionEngine,
}

impl ForkChoiceRule {
    pub async fn find_head(
        &self,
        current_slot: Slot,
        finalized_block: BlockHash,
        justified_checkpoint: Checkpoint,
        attestations: &[Attestation],
    ) -> Result<ForkChoiceResult, ForkChoiceError> {
        // Phase 1: Update block DAG with latest attestations
        self.update_block_dag(attestations, current_slot).await?;
        
        // Phase 2: Calculate LMD-GHOST weights
        let ghost_weights = self.lmd_ghost.calculate_weights(current_slot, &self.block_dag).await?;
        
        // Phase 3: Apply FFG voting rules
        let ffg_votes = self.ffg_voting.process_ffg_votes(attestations, justified_checkpoint).await?;
        
        // Phase 4: Combine LMD-GHOST and FFG results
        let combined_weights = self.combine_weights(ghost_weights, ffg_votes).await?;
        
        // Phase 5: Apply reorg protection
        let protected_weights = self.reorg_protector.apply_protection(combined_weights, finalized_block).await?;
        
        // Phase 6: Select chain head
        let head_selection = self.chain_selection.select_head(protected_weights, current_slot).await?;
        
        // Phase 7: Validate selection against safety rules
        self.validate_head_selection(&head_selection, finalized_block).await?;
        
        Ok(head_selection)
    }
    
    pub async fn handle_fork(
        &self,
        block_a: BlockHash,
        block_b: BlockHash,
        current_slot: Slot,
    ) -> Result<ForkResolution, ForkChoiceError> {
        // Phase 1: Analyze fork characteristics
        let fork_analysis = self.analyze_fork_characteristics(block_a, block_b).await?;
        
        // Phase 2: Calculate cumulative validator scores for each branch
        let branch_a_score = self.calculate_branch_score(block_a, current_slot).await?;
        let branch_b_score = self.calculate_branch_score(block_b, current_slot).await?;
        
        // Phase 3: Apply scoring rules with time-lived components
        let weighted_scores = self.apply_time_lived_scoring(branch_a_score, branch_b_score).await?;
        
        // Phase 4: Check for finality violations
        let finality_check = self.check_finality_violations(block_a, block_b).await?;
        
        // Phase 5: Apply economic incentives consideration
        let economic_analysis = self.analyze_economic_impact(block_a, block_b).await?;
        
        // Phase 6: Make fork resolution decision
        let resolution = self.resolve_fork(
            fork_analysis,
            weighted_scores,
            finality_check,
            economic_analysis,
        ).await?;
        
        // Phase 7: Update fork choice state
        self.update_fork_choice_state(resolution.clone()).await?;
        
        Ok(resolution)
    }
    
    async fn calculate_branch_score(
        &self,
        block_hash: BlockHash,
        current_slot: Slot,
    ) -> Result<BranchScore, ForkChoiceError> {
        let branch_blocks = self.block_dag.get_branch(block_hash).await?;
        
        // Calculate cumulative validator score for the branch
        let cumulative_score: f64 = branch_blocks.par_iter()
            .map(|block| {
                let validator_score = self.get_validator_score(block.proposer).await.unwrap_or(0.0);
                let time_decay = self.calculate_time_decay(block.slot, current_slot).await;
                validator_score * time_decay
            })
            .sum();
        
        // Calculate branch weight based on attestations
        let attestation_weight = self.calculate_attestation_weight(&branch_blocks).await?;
        
        // Calculate finality progress
        let finality_progress = self.calculate_finality_progress(&branch_blocks).await?;
        
        Ok(BranchScore {
            block_hash,
            cumulative_validator_score: cumulative_score,
            attestation_weight,
            finality_progress,
            branch_length: branch_blocks.len() as u64,
            average_time_lived: self.calculate_average_time_lived(&branch_blocks).await?,
            economic_weight: self.calculate_economic_weight(&branch_blocks).await?,
        })
    }
    
    async fn apply_time_lived_scoring(
        &self,
        branch_a: BranchScore,
        branch_b: BranchScore,
    ) -> Result<WeightedScores, ForkChoiceError> {
        // Calculate time-lived advantage for each branch
        let time_lived_advantage_a = self.calculate_time_lived_advantage(&branch_a).await?;
        let time_lived_advantage_b = self.calculate_time_lived_advantage(&branch_b).await?;
        
        // Apply nonlinear transformation to time-lived scores
        let transformed_advantage_a = time_lived_advantage_a.powf(self.fork_choice_config.time_lived_exponent);
        let transformed_advantage_b = time_lived_advantage_b.powf(self.fork_choice_config.time_lived_exponent);
        
        // Combine with validator scores
        let weighted_score_a = branch_a.cumulative_validator_score * transformed_advantage_a;
        let weighted_score_b = branch_b.cumulative_validator_score * transformed_advantage_b;
        
        // Apply economic weight multiplier
        let economic_multiplier_a = self.calculate_economic_multiplier(&branch_a).await?;
        let economic_multiplier_b = self.calculate_economic_multiplier(&branch_b).await?;
        
        let final_score_a = weighted_score_a * economic_multiplier_a;
        let final_score_b = weighted_score_b * economic_multiplier_b;
        
        Ok(WeightedScores {
            branch_a: final_score_a,
            branch_b: final_score_b,
            time_lived_advantage_a,
            time_lived_advantage_b,
            economic_multiplier_a,
            economic_multiplier_b,
            score_difference: (final_score_a - final_score_b).abs(),
        })
    }
    
    pub async fn validate_chain_selection(
        &self,
        selected_head: BlockHash,
        alternatives: &[BlockHash],
        current_state: &ConsensusState,
    ) -> Result<ChainSelectionValidation, ForkChoiceError> {
        // Phase 1: Check safety against finalized blocks
        let safety_check = self.validate_against_finalized(selected_head, current_state.finalized_checkpoint).await?;
        
        // Phase 2: Verify justification consistency
        let justification_check = self.verify_justification_consistency(selected_head, current_state.justified_checkpoint).await?;
        
        // Phase 3: Check for equivocation protection
        let equivocation_check = self.check_equivocation_protection(selected_head, alternatives).await?;
        
        // Phase 4: Validate economic rationality
        let economic_validation = self.validate_economic_rationality(selected_head, alternatives).await?;
        
        // Phase 5: Verify time-lived constraints
        let time_constraints = self.verify_time_constraints(selected_head, current_state.current_slot).await?;
        
        Ok(ChainSelectionValidation {
            valid: safety_check.pass && 
                   justification_check.consistent && 
                   !equivocation_check.equivocation_detected &&
                   economic_validation.rational &&
                   time_constraints.satisfied,
            selected_head,
            safety_violations: safety_check.violations,
            justification_issues: justification_check.issues,
            equivocation_warnings: equivocation_check.warnings,
            economic_concerns: economic_validation.concerns,
            time_constraint_violations: time_constraints.violations,
            overall_confidence: self.calculate_validation_confidence(
                &safety_check,
                &justification_check,
                &equivocation_check,
                &economic_validation,
                &time_constraints,
            ).await?,
        })
    }
}