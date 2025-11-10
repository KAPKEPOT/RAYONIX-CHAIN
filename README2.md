
RAYONIX-CHAIN

A complete blockchain implementation with Proof-of-Stake consensus, smart contracts, and UTXO system—built from scratch in Python.

https://img.shields.io/badge/Architecture-Modular-blue
https://img.shields.io/badge/Python-3.8%2B-green
https://img.shields.io/badge/License-AGPL--3.0-orange

🏗️ Modular Architecture

```
RAYONIX-CHAIN/
├── 🏗️  Core Modules
│   ├── blockchain/           # Blockchain core logic
│   ├── consensusengine/      # Proof-of-Stake consensus
│   ├── network/              # P2P networking
│   ├── database/             # Data persistence
│   └── merkle_system/        # Merkle tree implementation
│
├── 💰  Financial Modules
│   ├── rayonix_wallet/       # HD wallet management
│   ├── utxo_system/          # UTXO transaction model
│   └── smart_contract/       # Smart contract engine
│
├── 🖥️  Runtime
│   ├── rayonix_node/         # Node management
│   ├── rayonixd.py           # Main node daemon
│   ├── rayonix_cli.py        # Interactive CLI client
│   └── main.py               # Alternative entry point
│
├── ⚙️  Configuration
│   ├── config/               # Configuration management
│   ├── rayonix.yaml          # Node settings
│   └── pyproject.toml        # Package configuration
│
└── 📄  Documentation
    ├── README.md             # This file
    ├── LICENSE               # AGPL-3.0 License
    └── structure.txt         # Project structure
```

🚀 Get Started in 2 Minutes

1. Install & Setup

```bash
# Clone and enter
git clone https://github.com/RayoniR/RAYONIX-CHAIN.git
cd RAYONIX-CHAIN

# Install dependencies
pip install -r requirements.txt
```

2. Start Blockchain Node

```bash
# Terminal 1 - Run the node daemon
./rayonixd.py
```

Wait for node to start up and begin syncing

3. Use the Interactive Client

```bash
# Terminal 2 - Open the CLI client
./rayonix_cli.py
```

You'll see:

```
=============================
RAYONIX BLOCKCHAIN CLI
Connected to daemon via RPC
=============================
Type 'help' for available commands
Type 'exit' or 'quit' to exit
=============================
rayonix>
```

🎯 Quick Start Guide

First Steps in the CLI

```bash
rayonix> help                    # See all available commands
rayonix> create-wallet           # Create your first wallet
rayonix> balance                 # Check your balance
rayonix> info                    # See node status
```

📋 Complete Command Reference

👛 Wallet Commands

```bash
create-wallet      # Create a new HD wallet
load-wallet        # Load wallet from mnemonic phrase  
import-wallet      # Import wallet from backup file
backup-wallet      # Backup wallet to file
address            # Generate new address
list-addresses     # List all wallet addresses
balance            # Show detailed wallet balance
wallet-info        # Show wallet information
send               # Send coins to address
```

⛓️ Blockchain Commands

```bash
blockchain-info    # Show blockchain information
block              # Show block information
transaction        # Show transaction details
history            # Show transaction history
mempool            # Show mempool information
sync-status        # Show sync status
```

🌐 Network Commands

```bash
network            # Show network statistics
peers              # Show connected peers
```

⚡ Staking & Validation

```bash
stake              # Stake tokens for validation
staking            # Show staking information
validator-info     # Show validator information
```

🤖 Smart Contracts

```bash
deploy-contract    # Deploy smart contract
call-contract      # Call contract function
contracts          # List smart contracts
```

🏗️ Module Overview

Module Purpose Key Features
blockchain/ Core chain logic Block validation, chain organization
consensusengine/ PoS Consensus Validator selection, block finality
network/ P2P Networking Peer discovery, message propagation
utxo_system/ Transaction model UTXO management, transaction verification
rayonix_wallet/ Wallet management HD wallets, key derivation, signing
smart_contract/ Contract engine VM execution, contract deployment
merkle_system/ Data integrity Merkle trees, proof generation
database/ Storage Data persistence, efficient querying

🛠️ Development

Package Installation

```bash
# Install as editable package (recommended)
pip install -e .

# Use command-line tools
rayonix-node    # Start node daemon
rayonix-cli     # Start CLI client
```

Running Tests

```bash
# Run test suite
python -m pytest tests/ -v

# Security audit
python -m bandit -r ./
```

⚙️ Configuration

Edit rayonix.yaml to customize your node:

```yaml
network:
  port: 52555
  max_peers: 50
  
consensus:
  staking_enabled: true
  minimum_stake: 1000

database:
  path: ./rayonix_data
```

❓ Support

· 📚 Documentation: https://docs.rayonix.site
· 🐛 Issues: GitHub Issues
· 💬 Community: Discord

📜 License

AGPL-3.0 - See LICENSE for details.

---

Built with Python · Modular Architecture · Enterprise Ready

Start building decentralized applications today! 🚀

---
