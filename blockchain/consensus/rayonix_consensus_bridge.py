# blockchain/consensus/rayonix_consensus_bridge.py
import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ConsensusBridgeState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SYNCING = "syncing"
    READY = "ready"
    ERROR = "error"

@dataclass
class ConsensusMetrics:
    """Metrics from the Rust consensus engine"""
    epoch: int = 0
    slot: int = 0
    validator_count: int = 0
    total_stake: int = 0
    average_score: float = 0.0
    gini_coefficient: float = 0.0
    entropy: float = 0.0
    security_score: float = 0.0
    health_score: float = 0.0

@dataclass
class ValidatorInfo:
    """Validator information from consensus"""
    validator_id: str
    stake: int
    reliability_score: float
    total_score: float
    selection_probability: float
    status: str
    last_active_slot: int

class RayonixConsensusBridge:
    """
    Bridge between Python blockchain and Rust consensus engine
    Handles communication, data serialization, and state synchronization
    """
    
    def __init__(self, config: Dict[str, Any], data_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.state = ConsensusBridgeState.DISCONNECTED
        self.rust_process = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Communication pipes
        self.input_pipe = None
        self.output_pipe = None
        
        # State tracking
        self.current_epoch = 0
        self.current_slot = 0
        self.validators: Dict[str, ValidatorInfo] = {}
        self.metrics = ConsensusMetrics()
        
        # Threading
        self.lock = threading.RLock()
        self.callbacks = {
            'slot_processed': [],
            'epoch_transition': [],
            'validator_update': [],
            'consensus_anomaly': []
        }
        
    async def start(self):
        """Start the Rust consensus engine"""
        if self.state != ConsensusBridgeState.DISCONNECTED:
            logger.warning("Consensus bridge already running")
            return
            
        try:
            logger.info("Starting Rust consensus engine...")
            self.state = ConsensusBridgeState.CONNECTING
            
            # Build paths
            consensus_binary = self._find_consensus_binary()
            if not consensus_binary:
                raise RuntimeError("Consensus binary not found")
                
            # Create pipes for communication
            input_path = os.path.join(self.data_dir, "consensus_input.pipe")
            output_path = os.path.join(self.data_dir, "consensus_output.pipe")
            
            # Create named pipes
            self._create_pipes(input_path, output_path)
            
            # Start Rust process
            process_args = [
                consensus_binary,
                "--network", self.config.get('network_type', 'mainnet'),
                "--data-dir", self.data_dir,
                "--input-pipe", input_path,
                "--output-pipe", output_path,
                "--config", json.dumps(self.config)
            ]
            
            self.rust_process = await asyncio.create_subprocess_exec(
                *process_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Open communication pipes
            self.input_pipe = open(input_path, 'w')
            self.output_pipe = open(output_path, 'r')
            
            # Start background tasks
            asyncio.create_task(self._monitor_rust_process())
            asyncio.create_task(self._read_consensus_output())
            
            self.state = ConsensusBridgeState.CONNECTED
            logger.info("Rust consensus engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start consensus engine: {e}")
            self.state = ConsensusBridgeState.ERROR
            raise
            
    async def stop(self):
        """Stop the Rust consensus engine gracefully"""
        if self.state == ConsensusBridgeState.DISCONNECTED:
            return
            
        logger.info("Stopping Rust consensus engine...")
        
        # Send shutdown command
        await self._send_command("shutdown", {})
        
        # Wait for process to terminate
        if self.rust_process:
            try:
                await asyncio.wait_for(self.rust_process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Consensus process didn't terminate gracefully, forcing...")
                self.rust_process.terminate()
                
        # Close pipes
        if self.input_pipe:
            self.input_pipe.close()
        if self.output_pipe:
            self.output_pipe.close()
            
        self.state = ConsensusBridgeState.DISCONNECTED
        logger.info("Rust consensus engine stopped")
        
    async def process_slot(self, slot: int, parent_block_hash: str, 
                         validators: List[Dict[str, Any]],
                         network_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a slot through the Rust consensus engine
        """
        if self.state != ConsensusBridgeState.READY:
            raise RuntimeError("Consensus engine not ready")
            
        command = {
            "type": "process_slot",
            "slot": slot,
            "parent_block_hash": parent_block_hash,
            "validators": validators,
            "network_state": network_state
        }
        
        response = await self._send_command("process_slot", command)
        
        # Update local state
        with self.lock:
            self.current_slot = slot
            if 'leader_selection' in response:
                self._update_validators(response.get('validators', []))
                
        # Notify subscribers
        self._notify_callbacks('slot_processed', {
            'slot': slot,
            'leader': response.get('leader_selection', {}),
            'scores': response.get('scores', {})
        })
        
        return response
        
    async def process_epoch_transition(self, epoch: int, 
                                     network_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process epoch transition through Rust consensus
        """
        command = {
            "type": "process_epoch",
            "epoch": epoch,
            "network_state": network_state
        }
        
        response = await self._send_command("process_epoch", command)
        
        # Update local state
        with self.lock:
            self.current_epoch = epoch
            self.metrics = ConsensusMetrics(**response.get('metrics', {}))
            
        # Notify subscribers
        self._notify_callbacks('epoch_transition', {
            'epoch': epoch,
            'metrics': self.metrics
        })
        
        return response
        
    async def get_validator_info(self, validator_id: str) -> Optional[ValidatorInfo]:
        """Get validator information from consensus"""
        with self.lock:
            return self.validators.get(validator_id)
            
    async def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get current consensus metrics"""
        with self.lock:
            return self.metrics
            
    async def handle_fork(self, fork_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fork situation through consensus engine"""
        command = {
            "type": "handle_fork",
            "fork_data": fork_data
        }
        
        return await self._send_command("handle_fork", command)
        
    async def optimize_parameters(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize consensus parameters based on historical data"""
        command = {
            "type": "optimize_parameters",
            "historical_data": historical_data
        }
        
        return await self._send_command("optimize_parameters", command)
        
    # Internal methods
    
    def _find_consensus_binary(self) -> Optional[str]:
        """Find the Rust consensus binary"""
        possible_paths = [
            "./target/release/rayonix_consensus",
            "/usr/local/bin/rayonix_consensus",
            os.path.join(os.path.dirname(__file__), "../../consensus/target/release/rayonix_consensus")
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
                
        logger.error("Consensus binary not found in any expected location")
        return None
        
    def _create_pipes(self, input_path: str, output_path: str):
        """Create named pipes for communication"""
        try:
            # Remove existing pipes
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
                    
            # Create new pipes
            os.mkfifo(input_path)
            os.mkfifo(output_path)
            
            logger.info(f"Created communication pipes: {input_path}, {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create pipes: {e}")
            raise
            
    async def _send_command(self, command_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to Rust process and wait for response"""
        if not self.input_pipe or not self.output_pipe:
            raise RuntimeError("Communication pipes not open")
            
        # Add timestamp and ID for tracking
        command_id = str(hash(str(data) + str(asyncio.get_event_loop().time())))
        data['command_id'] = command_id
        data['timestamp'] = asyncio.get_event_loop().time()
        
        try:
            # Send command
            command_json = json.dumps(data) + '\n'
            self.input_pipe.write(command_json)
            self.input_pipe.flush()
            
            # Wait for response (simplified - in production you'd want proper async I/O)
            response_line = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.output_pipe.readline
            )
            
            if response_line:
                response = json.loads(response_line.strip())
                if response.get('command_id') == command_id:
                    return response
                else:
                    logger.warning(f"Response ID mismatch for command {command_type}")
                    
            return {}
            
        except Exception as e:
            logger.error(f"Error sending command {command_type}: {e}")
            return {}
            
    async def _read_consensus_output(self):
        """Continuously read output from consensus engine"""
        while self.state in [ConsensusBridgeState.CONNECTED, ConsensusBridgeState.READY]:
            try:
                if self.output_pipe:
                    line = await asyncio.get_event_loop().run_in_executor(
                        self.executor, self.output_pipe.readline
                    )
                    
                    if line:
                        data = json.loads(line.strip())
                        await self._handle_consensus_message(data)
                    else:
                        await asyncio.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"Error reading consensus output: {e}")
                await asyncio.sleep(1.0)
                
    async def _handle_consensus_message(self, data: Dict[str, Any]):
        """Handle incoming messages from consensus engine"""
        msg_type = data.get('type')
        
        if msg_type == 'ready':
            self.state = ConsensusBridgeState.READY
            logger.info("Consensus engine is ready")
            
        elif msg_type == 'validator_update':
            self._update_validators(data.get('validators', []))
            self._notify_callbacks('validator_update', data)
            
        elif msg_type == 'consensus_anomaly':
            logger.warning(f"Consensus anomaly detected: {data.get('anomaly_type')}")
            self._notify_callbacks('consensus_anomaly', data)
            
        elif msg_type == 'metrics_update':
            with self.lock:
                self.metrics = ConsensusMetrics(**data.get('metrics', {}))
                
        elif msg_type == 'error':
            logger.error(f"Consensus engine error: {data.get('error')}")
            self.state = ConsensusBridgeState.ERROR
            
    def _update_validators(self, validators_data: List[Dict[str, Any]]):
        """Update local validator cache"""
        with self.lock:
            self.validators.clear()
            for validator_data in validators_data:
                validator = ValidatorInfo(**validator_data)
                self.validators[validator.validator_id] = validator
                
    def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")
                
    def subscribe(self, event_type: str, callback: callable):
        """Subscribe to consensus events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)