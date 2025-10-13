// consensus/src/main.rs
use rayonix_consensus::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let config = parse_arguments(&args)?;
    
    // Initialize logging
    init_logging(&config)?;
    
    // Create consensus engine
    let mut consensus_engine = HybridOrchestrator::new(config.consensus_config).await?;
    
    // Setup communication pipes
    let (command_tx, command_rx) = mpsc::channel(100);
    let (response_tx, response_rx) = mpsc::channel(100);
    
    // Start pipe readers/writers
    tokio::spawn(pipe_reader(config.input_pipe.clone(), command_tx));
    tokio::spawn(pipe_writer(config.output_pipe.clone(), response_rx));
    
    // Start command processor
    tokio::spawn(command_processor(
        consensus_engine, 
        command_rx, 
        response_tx
    ));
    
    // Send ready signal
    let ready_msg = json!({
        "type": "ready",
        "timestamp": std::time::SystemTime::now()
    });
    
    // Keep main thread alive
    tokio::signal::ctrl_c().await?;
    Ok(())
}

async fn command_processor(
    mut engine: HybridOrchestrator,
    mut command_rx: mpsc::Receiver<Value>,
    response_tx: mpsc::Sender<Value>,
) {
    while let Some(command) = command_rx.recv().await {
        let response = process_command(&mut engine, command).await;
        let _ = response_tx.send(response).await;
    }
}

async fn process_command(engine: &mut HybridOrchestrator, command: Value) -> Value {
    let command_type = command["type"].as_str().unwrap_or("unknown");
    
    match command_type {
        "process_slot" => {
            let slot = command["slot"].as_u64().unwrap_or(0);
            let parent_hash = command["parent_block_hash"].as_str().unwrap_or("");
            let validators = parse_validators(&command["validators"]);
            let network_state = parse_network_state(&command["network_state"]);
            
            let result = engine.execute_consensus_round(
                slot, parent_hash, &validators, &network_state
            ).await;
            
            json!({
                "command_id": command["command_id"],
                "type": "slot_result",
                "result": result
            })
        }
        "shutdown" => {
            json!({
                "command_id": command["command_id"],
                "type": "shutdown_ack"
            })
        }
        _ => {
            json!({
                "command_id": command["command_id"],
                "type": "error",
                "error": format!("Unknown command: {}", command_type)
            })
        }
    }
}