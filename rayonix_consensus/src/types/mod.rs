// src/types/mod.rs

pub mod validator;
pub mod score;
pub mod election;
pub mod slashing;
pub mod consensus;

pub fn register_type_definitions() {
    println!("Consensus type definitions loaded.");
}