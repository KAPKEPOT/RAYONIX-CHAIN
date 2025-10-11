#!/usr/bin/env python3
# main.py - Entry point and CLI interface for RAYONIX blockchain node

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from rayonix_node.core.node import RayonixNode
from rayonix_node.cli.interactive import run_interactive_mode
from rayonix_node.cli.command_handler import setup_cli_handlers
from rayonix_node.utils.helpers import configure_logging

logger = logging.getLogger("rayonix_node")

async def main():
    """Main function with enhanced CLI using argparse"""
    parser = argparse.ArgumentParser(description='RAYONIX Blockchain Node')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--network', '-n', choices=['mainnet', 'testnet', 'regtest'], 
                       help='Network type')
    parser.add_argument('--port', '-p', type=int, help='P2P network port')
    parser.add_argument('--api-port', type=int, help='API server port')
    parser.add_argument('--no-api', action='store_true', help='Disable API server')
    parser.add_argument('--no-network', action='store_true', help='Disable P2P network')
    parser.add_argument('--data-dir', help='Data directory path')
    parser.add_argument('--encryption-key', help='Configuration encryption key')
    parser.add_argument('--wallet-mnemonic', help='Wallet mnemonic phrase')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Start in interactive mode (default)')
    parser.add_argument('--daemon', '-d', action='store_true', 
                       help='Start as daemon (non-interactive)')
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging()
    
    # Create node instance
    node = RayonixNode()
    
    # Initialize components
    if not await node.initialize_components(args.config, args.encryption_key):
        logger.error("Failed to initialize node components")
        return 1
    
    # Override config with command line arguments if provided
    if args.network:
        node.config_manager.set('network.network_type', args.network)
    
    if args.port:
        node.config_manager.set('network.listen_port', args.port)
    
    if args.api_port:
        node.config_manager.set('api.port', args.api_port)
    
    if args.no_api:
        node.config_manager.set('api.enabled', False)
    
    if args.no_network:
        node.config_manager.set('network.enabled', False)
    
    if args.data_dir:
        node.config_manager.set('database.db_path', args.data_dir)
    
    # Load wallet from mnemonic if provided
    if args.wallet_mnemonic and node.wallet is None:
        try:
            from rayonix_wallet.core.wallet import RayonixWallet
            node.wallet = RayonixWallet()
            if node.wallet.restore_from_mnemonic(args.wallet_mnemonic):
                logger.info("Wallet loaded from command line mnemonic")
                # Set blockchain reference
                if node.wallet.set_blockchain_reference(node.rayonix_chain):
                    logger.info("Wallet blockchain integration established")
                else:
                    logger.warning("Wallet loaded but blockchain integration failed")
            else:
                logger.error("Failed to load wallet from mnemonic")
        except Exception as e:
            logger.error(f"Error loading wallet from mnemonic: {e}")
    
    # Start the node
    if not await node.start():
        logger.error("Failed to start node")
        return 1
    
    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(node.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run in appropriate mode
    if args.daemon:
        logger.info("Running in daemon mode")
        try:
            # Keep the node running until shutdown signal
            while node.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
    else:
        # Interactive mode
        await run_interactive_mode(node)
    
    # Ensure node is stopped
    if node.running:
        await node.stop()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)