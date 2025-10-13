// consensus/chemistry/reaction_network.rs
use crate::types::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;
use nalgebra::{DVector, DMatrix, Matrix3, LU, SVD};
use petgraph::{Graph, Directed, graph::NodeIndex};
use petgraph::algo::{toposort, has_path_connecting};
use statrs::distribution::{Normal, Continuous};

pub struct ChemicalReactionNetwork {
    reaction_database: ReactionDatabase,
    stoichiometry_matrix: StoichiometryMatrix,
    rate_equation_solver: RateEquationSolver,
    equilibrium_analyzer: EquilibriumAnalyzer,
    metabolic_pathways: MetabolicPathwayAnalyzer,
    enzyme_kinetics: EnzymeKineticsEngine,
    chemical_thermodynamics: ChemicalThermodynamics,
}

impl ChemicalReactionNetwork {
    pub async fn model_consensus_as_reaction_network(
        &self,
        validators: &[ActiveValidator],
        network_state: &NetworkState,
    ) -> Result<ReactionNetworkModel, ChemistryError> {
        // Phase 1: Construct reaction graph from validator interactions
        let reaction_graph = self.build_reaction_graph(validators, network_state).await?;
        
        // Phase 2: Build stoichiometry matrix for mass balance
        let stoichiometry_matrix = self.build_stoichiometry_matrix(&reaction_graph).await?;
        
        // Phase 3: Solve system of differential equations for reaction kinetics
        let kinetic_solution = self.solve_reaction_kinetics(&reaction_graph, network_state).await?;
        
        // Phase 4: Analyze network stability and equilibrium
        let stability_analysis = self.analyze_network_stability(&stoichiometry_matrix, &kinetic_solution).await?;
        
        // Phase 5: Identify critical metabolic pathways
        let metabolic_pathways = self.identify_metabolic_pathways(&reaction_graph, validators).await?;
        
        // Phase 6: Apply enzyme kinetics to reaction rates
        let enzyme_kinetics = self.apply_enzyme_kinetics(&reaction_graph, &kinetic_solution).await?;
        
        // Phase 7: Calculate thermodynamic properties
        let thermodynamics = self.calculate_thermodynamic_properties(&reaction_graph, &kinetic_solution, network_state).await?;

        Ok(ReactionNetworkModel {
            reaction_graph,
            stoichiometry_matrix,
            kinetic_solution,
            stability_analysis,
            metabolic_pathways,
            enzyme_kinetics,
            thermodynamics,
            mass_balance: self.verify_mass_balance(&stoichiometry_matrix).await?,
            flux_analysis: self.perform_flux_analysis(&reaction_graph).await?,
        })
    }
    
    async fn build_reaction_graph(
        &self,
        validators: &[ActiveValidator],
        network_state: &NetworkState,
    ) -> Result<ReactionGraph, ChemistryError> {
        let mut graph = Graph::<ReactionSpecies, ReactionEdge, Directed>::new();
        
        // Create nodes for each validator as chemical species
        let species_nodes: HashMap<ValidatorId, NodeIndex> = validators
            .par_iter()
            .map(|validator| {
                let species = ReactionSpecies::Validator {
                    id: validator.identity.id,
                    concentration: validator.current_score,
                    chemical_potential: self.calculate_chemical_potential(validator).await?,
                    molecular_weight: self.calculate_validator_molecular_weight(validator).await?,
                    oxidation_state: self.calculate_oxidation_state(validator).await?,
                };
                let node_idx = graph.add_node(species);
                Ok((validator.identity.id, node_idx))
            })
            .collect::<Result<HashMap<_, _>, ChemistryError>>()?;
        
        // Add network states as additional species
        let network_species = self.create_network_species(network_state).await?;
        let network_nodes: HashMap<NetworkSpeciesType, NodeIndex> = network_species
            .into_iter()
            .map(|species| {
                let node_idx = graph.add_node(species.clone());
                (species.species_type(), node_idx)
            })
            .collect();
        
        // Create reactions between validators using combinatorial chemistry
        let reactions = self.generate_validator_reactions(validators, network_state).await?;
        
        for reaction in reactions {
            let source_node = species_nodes[&reaction.source_validator];
            let target_node = species_nodes[&reaction.target_validator];
            
            let reaction_edge = ReactionEdge {
                id: reaction.id,
                rate_constant: reaction.rate_constant,
                activation_energy: reaction.activation_energy,
                stoichiometry_coefficient: reaction.stoichiometry,
                reactant_order: reaction.reactant_order,
                product_order: reaction.product_order,
                enzyme_catalyzed: reaction.enzyme_catalyzed,
                michaelis_constant: reaction.michaelis_constant,
                max_rate: reaction.max_rate,
                reaction_type: reaction.reaction_type,
                thermodynamic_properties: self.calculate_reaction_thermodynamics(&reaction).await?,
            };
            
            graph.add_edge(source_node, target_node, reaction_edge.clone());
            
            // Add reverse reaction for reversible processes
            if reaction.reversible {
                let reverse_edge = reaction_edge.reverse_reaction().await?;
                graph.add_edge(target_node, source_node, reverse_edge);
            }
        }
        
        // Add network-wide catalytic reactions
        self.add_network_catalytic_reactions(&mut graph, &species_nodes, &network_nodes, network_state).await?;
        
        Ok(ReactionGraph {
            graph,
            species_nodes,
            network_nodes,
            reaction_count: graph.edge_count() as u32,
            connectivity: self.calculate_graph_connectivity(&graph).await?,
            diameter: self.calculate_graph_diameter(&graph).await?,
            clustering_coefficient: self.calculate_clustering_coefficient(&graph).await?,
        })
    }
    
    async fn build_stoichiometry_matrix(
        &self,
        reaction_graph: &ReactionGraph,
    ) -> Result<StoichiometryMatrix, ChemistryError> {
        let n_species = reaction_graph.graph.node_count();
        let n_reactions = reaction_graph.graph.edge_count();
        
        let mut stoichiometry_matrix = DMatrix::zeros(n_species, n_reactions);
        let mut reaction_metadata = Vec::with_capacity(n_reactions);

        // Build stoichiometry matrix: S_ij = stoichiometric coefficient of species i in reaction j
        for (reaction_idx, edge) in reaction_graph.graph.raw_edges().iter().enumerate() {
            let source_species = &reaction_graph.graph[edge.source()];
            let target_species = &reaction_graph.graph[edge.target()];
            let reaction = &edge.weight;
            
            // Reactants have negative coefficients, products positive
            stoichiometry_matrix[(edge.source().index(), reaction_idx)] = 
                -reaction.stoichiometry_coefficient * reaction.reactant_order;
            stoichiometry_matrix[(edge.target().index(), reaction_idx)] = 
                reaction.stoichiometry_coefficient * reaction.product_order;
            
            // Store reaction metadata for flux analysis
            reaction_metadata.push(ReactionMetadata {
                id: reaction.id,
                source_species: source_species.species_id(),
                target_species: target_species.species_id(),
                reaction_type: reaction.reaction_type.clone(),
                thermodynamic_parameters: reaction.thermodynamic_properties.clone(),
            });
        }
        
        // Perform singular value decomposition for pathway analysis
        let svd = SVD::new(stoichiometry_matrix.clone(), true, true);
        
        // Calculate conservation relations (left null space)
        let conservation_matrix = self.calculate_conservation_relations(&stoichiometry_matrix).await?;
        
        // Verify mass balance: S · v = 0 for steady state
        self.verify_stoichiometric_consistency(&stoichiometry_matrix).await?;
        
        Ok(StoichiometryMatrix {
            matrix: stoichiometry_matrix,
            species_count: n_species,
            reaction_count: n_reactions,
            null_space: svd.v_t.map(|v| v.transpose()).unwrap_or(DMatrix::zeros(n_reactions, 0)),
            left_null_space: conservation_matrix,
            reaction_metadata,
            rank: svd.rank(1e-10),
            singular_values: svd.singular_values.as_slice().to_vec(),
        })
    }
    
    async fn solve_reaction_kinetics(
        &self,
        reaction_graph: &ReactionGraph,
        network_state: &NetworkState,
    ) -> Result<KineticSolution, ChemistryError> {
        let n_species = reaction_graph.graph.node_count();
        let time_steps = 2000;
        let dt = 0.05;
        
        // Initialize concentrations with Boltzmann distribution
        let mut concentrations = DVector::from_fn(n_species, |i, _| {
            let species = &reaction_graph.graph[NodeIndex::new(i)];
            species.concentration() * (-species.chemical_potential() / (BOLTZMANN_CONSTANT * network_state.temperature)).exp()
        });
        
        let mut concentration_history = Vec::with_capacity(time_steps as usize);
        concentration_history.push(concentrations.clone());
        
        let mut reaction_rate_history = Vec::new();
        
        // Solve system of ODEs using 4th order Runge-Kutta method
        for step in 0..time_steps {
            let k1 = self.calculate_concentration_gradient(&concentrations, reaction_graph, network_state).await?;
            let k2 = self.calculate_concentration_gradient(&(&concentrations + 0.5 * dt * &k1), reaction_graph, network_state).await?;
            let k3 = self.calculate_concentration_gradient(&(&concentrations + 0.5 * dt * &k2), reaction_graph, network_state).await?;
            let k4 = self.calculate_concentration_gradient(&(&concentrations + dt * &k3), reaction_graph, network_state).await?;
            
            // RK4 update: C_{n+1} = C_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
            concentrations += (dt / 6.0) * (k1 + 2.0 * &k2 + 2.0 * &k3 + k4);
            
            // Apply physical constraints: non-negative concentrations
            concentrations.iter_mut().for_each(|c| *c = c.max(0.0));
            
            // Calculate current reaction rates for monitoring
            let current_rates = self.calculate_reaction_rates(&concentrations, reaction_graph, network_state).await?;
            reaction_rate_history.push(current_rates.clone());
            
            concentration_history.push(concentrations.clone());
            
            // Check for chemical equilibrium using detailed balance
            if self.check_chemical_equilibrium(&concentrations, &current_rates, reaction_graph).await? {
                break;
            }
        }
        
        // Calculate thermodynamic driving forces
        let affinity = self.calculate_reaction_affinity(&concentrations, reaction_graph).await?;
        let entropy_production = self.calculate_entropy_production(&reaction_rate_history, &affinity, network_state).await?;
        
        Ok(KineticSolution {
            concentration_history,
            final_concentrations: concentrations,
            reaction_rate_history,
            reaction_affinity: affinity,
            entropy_production,
            time_to_equilibrium: self.calculate_equilibrium_time(&concentration_history).await?,
            dynamic_stability: self.analyze_dynamic_stability(&concentration_history).await?,
            lyapunov_exponents: self.calculate_lyapunov_exponents(&concentration_history).await?,
        })
    }
    
    async fn calculate_concentration_gradient(
        &self,
        concentrations: &DVector<f64>,
        reaction_graph: &ReactionGraph,
        network_state: &NetworkState,
    ) -> Result<DVector<f64>, ChemistryError> {
        let reaction_rates = self.calculate_reaction_rates(concentrations, reaction_graph, network_state).await?;
        let flux_vector = self.calculate_flux_vector(&reaction_rates, reaction_graph).await?;
        
        // Fundamental equation: dC/dt = S · v
        let gradient = &reaction_graph.stoichiometry_matrix.matrix * &flux_vector;
        
        Ok(gradient)
    }
    
    async fn calculate_reaction_rates(
        &self,
        concentrations: &DVector<f64>,
        reaction_graph: &ReactionGraph,
        network_state: &NetworkState,
    ) -> Result<Vec<ReactionRate>, ChemistryError> {
        let rates: Vec<ReactionRate> = reaction_graph.graph.raw_edges()
            .par_iter()
            .map(|edge| {
                let source_conc = concentrations[edge.source().index()];
                let target_conc = concentrations[edge.target().index()];
                let reaction = &edge.weight;
                
                // Generalized mass action kinetics with activity coefficients
                let activity_source = self.calculate_activity_coefficient(source_conc, reaction).await?;
                let activity_target = self.calculate_activity_coefficient(target_conc, reaction).await?;
                
                let mass_action_rate = reaction.rate_constant * 
                    (activity_source * source_conc).powf(reaction.reactant_order) * 
                    (activity_target * target_conc).powf(reaction.product_order);
                
                // Michaelis-Menten kinetics for enzyme-catalyzed reactions
                let enzyme_rate = if reaction.enzyme_catalyzed {
                    self.calculate_michaelis_menten_rate(
                        source_conc,
                        reaction.michaelis_constant,
                        reaction.max_rate,
                        reaction.thermodynamic_properties.delta_g
                    ).await?
                } else {
                    0.0
                };
                
                // Arrhenius temperature dependence with pre-exponential factor
                let arrhenius_factor = reaction.rate_constant * 
                    (-reaction.activation_energy / (GAS_CONSTANT * network_state.temperature)).exp();
                
                // Transition state theory correction
                let transition_state_correction = self.calculate_transition_state_correction(reaction, network_state).await?;
                
                let total_rate = (mass_action_rate + enzyme_rate) * arrhenius_factor * transition_state_correction;
                
                // Calculate reaction affinity (thermodynamic driving force)
                let affinity = self.calculate_single_reaction_affinity(source_conc, target_conc, reaction).await?;
                
                Ok(ReactionRate {
                    rate: total_rate,
                    mass_action_component: mass_action_rate,
                    enzyme_component: enzyme_rate,
                    temperature_factor: arrhenius_factor,
                    transition_state_factor: transition_state_correction,
                    affinity,
                    reaction_id: reaction.id,
                    flux_control_coefficient: self.calculate_flux_control_coefficient(total_rate, concentrations).await?,
                })
            })
            .collect::<Result<Vec<_>, ChemistryError>>()?;
        
        Ok(rates)
    }
    
    async fn calculate_michaelis_menten_rate(
        &self,
        substrate_conc: f64,
        km: f64,
        vmax: f64,
        delta_g: f64,
    ) -> Result<f64, ChemistryError> {
        // Standard Michaelis-Menten equation with thermodynamic corrections
        let base_rate = (vmax * substrate_conc) / (km + substrate_conc);
        
        // Apply thermodynamic driving force correction
        let thermodynamic_factor = (-delta_g / (GAS_CONSTANT * 298.15)).exp().min(1e6).max(1e-6);
        
        Ok(base_rate * thermodynamic_factor)
    }
    
    async fn calculate_thermodynamic_properties(
        &self,
        reaction_graph: &ReactionGraph,
        kinetic_solution: &KineticSolution,
        network_state: &NetworkState,
    ) -> Result<ChemicalThermodynamics, ChemistryError> {
        let concentrations = &kinetic_solution.final_concentrations;
        
        // Calculate chemical potentials for all species
        let chemical_potentials: Vec<f64> = (0..reaction_graph.graph.node_count())
            .map(|i| {
                let species = &reaction_graph.graph[NodeIndex::new(i)];
                let standard_potential = species.chemical_potential();
                let concentration = concentrations[i];
                
                // μ = μ° + RT ln(a) where activity a = γ·C
                standard_potential + GAS_CONSTANT * network_state.temperature * concentration.ln().max(-100.0)
            })
            .collect();
        
        // Calculate reaction Gibbs free energy changes
        let delta_g_reactions: Vec<f64> = reaction_graph.graph.raw_edges()
            .iter()
            .map(|edge| {
                let mu_source = chemical_potentials[edge.source().index()];
                let mu_target = chemical_potentials[edge.target().index()];
                let stoichiometry = edge.weight.stoichiometry_coefficient;
                
                // ΔG = Σμ_products - Σμ_reactants
                stoichiometry * (mu_target - mu_source)
            })
            .collect();
        
        // Calculate overall network thermodynamics
        let total_entropy_production = kinetic_solution.entropy_production.iter().sum::<f64>();
        let total_affinity = kinetic_solution.reaction_affinity.iter().sum::<f64>();
        
        Ok(ChemicalThermodynamics {
            chemical_potentials,
            delta_g_reactions,
            total_entropy_production,
            total_affinity,
            dissipation_function: self.calculate_dissipation_function(&delta_g_reactions, &kinetic_solution.reaction_rate_history).await?,
            thermodynamic_efficiency: self.calculate_thermodynamic_efficiency(total_entropy_production, total_affinity).await?,
            far_from_equilibrium_measure: self.calculate_far_from_equilibrium_measure(&delta_g_reactions).await?,
        })
    }
}

// Physical constants for chemical kinetics
const GAS_CONSTANT: f64 = 8.314462618; // J/(mol·K)
const BOLTZMANN_CONSTANT: f64 = 1.380649e-23; // J/K
const AVOGADRO_CONSTANT: f64 = 6.02214076e23; // mol⁻¹
const FARADAY_CONSTANT: f64 = 96485.33212; // C/mol

#[derive(Debug, Clone)]
pub struct ReactionSpecies {
    pub id: SpeciesId,
    pub concentration: f64,
    pub chemical_potential: f64,
    pub molecular_weight: f64,
    pub oxidation_state: f64,
    pub species_type: SpeciesType,
}

impl ReactionSpecies {
    pub fn concentration(&self) -> f64 {
        self.concentration.max(0.0)
    }
    
    pub fn chemical_potential(&self) -> f64 {
        self.chemical_potential
    }
    
    pub fn species_id(&self) -> SpeciesId {
        self.id
    }
    
    pub fn species_type(&self) -> SpeciesType {
        self.species_type
    }
}

#[derive(Debug, Clone)]
pub struct ReactionEdge {
    pub id: ReactionId,
    pub rate_constant: f64,
    pub activation_energy: f64,
    pub stoichiometry_coefficient: f64,
    pub reactant_order: f64,
    pub product_order: f64,
    pub enzyme_catalyzed: bool,
    pub michaelis_constant: f64,
    pub max_rate: f64,
    pub reaction_type: ReactionType,
    pub thermodynamic_properties: ReactionThermodynamics,
}

impl ReactionEdge {
    pub async fn reverse_reaction(&self) -> Result<Self, ChemistryError> {
        let reverse_thermo = ReactionThermodynamics {
            delta_g: -self.thermodynamic_properties.delta_g,
            delta_h: -self.thermodynamic_properties.delta_h,
            delta_s: -self.thermodynamic_properties.delta_s,
            equilibrium_constant: 1.0 / self.thermodynamic_properties.equilibrium_constant,
        };
        
        Ok(ReactionEdge {
            id: format!("{}_reverse", self.id),
            rate_constant: self.calculate_reverse_rate_constant().await?,
            activation_energy: self.activation_energy + self.thermodynamic_properties.delta_g,
            stoichiometry_coefficient: self.stoichiometry_coefficient,
            reactant_order: self.product_order,
            product_order: self.reactant_order,
            enzyme_catalyzed: self.enzyme_catalyzed,
            michaelis_constant: self.calculate_reverse_michaelis_constant().await?,
            max_rate: self.max_rate,
            reaction_type: self.reaction_type.reverse(),
            thermodynamic_properties: reverse_thermo,
        })
    }
    
    async fn calculate_reverse_rate_constant(&self) -> Result<f64, ChemistryError> {
        // From detailed balance: k_reverse = k_forward / K_eq
        Ok(self.rate_constant / self.thermodynamic_properties.equilibrium_constant)
    }
    
    async fn calculate_reverse_michaelis_constant(&self) -> Result<f64, ChemistryError> {
        // For reversible Michaelis-Menten kinetics
        Ok(self.michaelis_constant * self.thermodynamic_properties.equilibrium_constant)
    }
}