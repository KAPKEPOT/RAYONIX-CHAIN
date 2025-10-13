// src/core/validator/mod.rs

pub mod registry;
pub mod state_manager;
pub mod lifecycle;

pub fn init_validator_system() {
    println!("Validator system initialized.");
}