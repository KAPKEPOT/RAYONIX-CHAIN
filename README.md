
# RAYONIX-CHAIN

A complete blockchain implementation with Proof-of-Stake consensus, smart contracts, and UTXO system—built from scratch in Python.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-AGPL--3.0-orange)
![Version](https://img.shields.io/badge/Version-0.1.0-green)
![Consensus](https://img.shields.io/badge/Consensus-Proof_of_Stake-success)
![Smart Contracts](https://img.shields.io/badge/Smart_Contracts-Enabled-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-success)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/KAPKEPOT/RAYONIX-CHAIN)





### 🚀 Get Started in 2 Minutes

1. **Install & Setup**

bash
 Clone and enter
```
git clone https://github.com/KAPKEPOT/RAYONIX-CHAIN.git
cd RAYONIX-CHAIN
```

###### Install dependencies
```
pip install -r requirements.txt
```

**2. Start Blockchain Node**

***bash***
###### Terminal 1 - Run the node daemon
```
./rayonixd.py
```

Wait for node to start up and begin syncing

**3. Use the Interactive Client**

***bash***
##### Terminal 2 - Open the CLI client
```
./rayonix_cli.py
```

You'll see the RAYONIX interface:

```
=============================
RAYONIX BLOCKCHAIN CLI
Connected to daemon via RPC
=============================
Node Status: Unknown
Block Height: 0
Connected Peers: 0

Type 'help' for available commands
Type 'exit' or 'quit' to exit
=============================
RAYONIX Blockchain CLI
Type 'help' for available commands
rayonix>
```

#### 🎯 Quick Start Guide

***First Steps in the CLI***

```
rayonix> help                    # See all available commands
rayonix> create-wallet           # Create your first wallet
rayonix> balance                 # Check your balance
rayonix> info                    # See node status
```

### 🆘 Getting Help

```
rayonix> help                    # Show all commands by category
rayonix> help send               # Get detailed help for specific command
```

### 📋 Complete Command Reference

###### 👛 Wallet Commands

```
create-wallet      # Create a new wallet
load-wallet        # Load wallet from mnemonic phrase  
import-wallet      # Import wallet from backup file
backup-wallet      # Backup wallet to file
address            # Generate new address
list-addresses     # List all wallet addresses
balance            # Show detailed wallet balance
wallet-info        # Show detailed wallet information
send               # Send coins to address
```

###### ⛓️ Blockchain Commands

```
blockchain-info    # Show detailed blockchain information
block              # Show block information
transaction        # Show transaction information
history            # Show transaction history
mempool            # Show mempool information
sync-status        # Show synchronization status
```

###### 🌐 Network Commands

```
network            # Show network statistics
peers              # Show connected peers with details
```

###### 🖥️ Node Commands

```
info               # Show detailed node information
status             # Show node status
```

###### ⚡ Staking & Validation

```
stake              # Stake tokens for validation
staking            # Show staking information
validator-info     # Show validator information
```

###### 🤖 Smart Contracts

```
deploy-contract    # Deploy smart contract
call-contract      # Call contract function
contracts          # List smart contracts
```

###### ⚙️ System Commands

```
config             # Show configuration information
stats              # Show CLI statistics
```

###### 🛠️ For Developers

###### Project Structure


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

#### Package Installation

***bash***
###### Install as editable package
```
pip install -e .
```

###### Use command-line tools (after installation)
```
rayonix-node    # Start node daemon
```
```
rayonix-cli     # Start CLI client
```

##### ❓ Troubleshooting

###### Common Issues

· ***"Node Status***: Unknown": Ensure rayonixd.py is running in another terminal
· ***"Connected Peers***: 0": Node is still starting up or firewall blocking connections
· ***Command not found***: Make sure you're in the RAYONIX-CHAIN directory

##### Getting Support

· ***📚 Documentation**: https://docs.rayonix.site
· 🐛 Report Issues: GitHub Issues
· 💬 Community: Discord

##### 📜 License

AGPL-3.0 - See LICENSE for details.

Ready to explore? Start with create-wallet and balance to begin your RAYONIX journey! 🚀

Built with Python · Proof-of-Stake · Smart Contracts · Open Source

---
