// src/security/slashing/mod.rs

pub mod detector;
pub mod penalty_calculator;
pub mod jail_manager;

pub fn start_slashing_protection() {
    println!("Slashing protection active.");
}