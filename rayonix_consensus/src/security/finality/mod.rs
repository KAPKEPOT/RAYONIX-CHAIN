// src/security/finality/mod.rs

pub mod gadget;
pub mod vote_aggregator;
pub mod fork_choice;

pub fn init_finality_gadget() {
    println!("Finality gadget initialized.");
}