
RAYONIX-CHAIN

A complete blockchain implementation with Proof-of-Stake consensus, smart contracts, and UTXO system—built from scratch in Python.

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

🎯 Quick Start Guide

First Steps in the CLI

```
rayonix> help                    # See all available commands
rayonix> create-wallet           # Create your first wallet
rayonix> balance                 # Check your balance
rayonix> info                    # See node status
```

🆘 Getting Help

```
rayonix> help                    # Show all commands by category
rayonix> help send               # Get detailed help for specific command
```

📋 Complete Command Reference

👛 Wallet Commands

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

⛓️ Blockchain Commands

```
blockchain-info    # Show detailed blockchain information
block              # Show block information
transaction        # Show transaction information
history            # Show transaction history
mempool            # Show mempool information
sync-status        # Show synchronization status
```

🌐 Network Commands

```
network            # Show network statistics
peers              # Show connected peers with details
```

🖥️ Node Commands

```
info               # Show detailed node information
status             # Show node status
```

⚡ Staking & Validation

```
stake              # Stake tokens for validation
staking            # Show staking information
validator-info     # Show validator information
```

🤖 Smart Contracts

```
deploy-contract    # Deploy smart contract
call-contract      # Call contract function
contracts          # List smart contracts
```

⚙️ System Commands

```
config             # Show configuration information
stats              # Show CLI statistics
```

🛠️ For Developers

Project Structure

```
RAYONIX-CHAIN/
├── rayonixd.py           # Main node daemon (RPC server)
├── rayonix_cli.py        # Interactive CLI client
├── main.py               # Alternative entry point
├── smart_contract/       # Smart contract engine
├── utxo_system/          # UTXO transaction system
├── rayonix.yaml          # Configuration file
└── requirements.txt      # Python dependencies
```

Package Installation

```bash
# Install as editable package
pip install -e .

# Use command-line tools (after installation)
rayonix-node    # Start node daemon
rayonix-cli     # Start CLI client
```

❓ Troubleshooting

Common Issues

· "Node Status: Unknown": Ensure rayonixd.py is running in another terminal
· "Connected Peers: 0": Node is still starting up or firewall blocking connections
· Command not found: Make sure you're in the RAYONIX-CHAIN directory

Getting Support

· 📚 Documentation: https://docs.rayonix.site
· 🐛 Report Issues: GitHub Issues
· 💬 Community: Discord

📜 License

AGPL-3.0 - See LICENSE for details.

---

Ready to explore? Start with create-wallet and balance to begin your RAYONIX journey! 🚀

Built with Python · Proof-of-Stake · Smart Contracts · Open Source

---
