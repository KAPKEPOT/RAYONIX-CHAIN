#!/usr/bin/env python3
"""
Wallet Management Screen
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import (
    Static, Button, DataTable, Input, Select, 
    Label, Markdown, Pretty
)
from textual.screen import Screen
from textual import events

class WalletScreen(Screen):
    """Comprehensive Wallet Management"""
    
    CSS = """
    WalletScreen {
        align: center middle;
        background: #0f0f23;
    }
    
    #wallet-container {
        grid-size: 2 2;
        grid-gutter: 1 2;
        padding: 1 2;
        height: 100%;
    }
    
    .wallet-section {
        background: #1a1b26;
        border: round #2a2b3c;
        padding: 1 2;
    }
    
    .balance-display {
        background: #16161e;
        border: double #7aa2f7;
        padding: 2;
        text-align: center;
    }
    
    .balance-amount {
        color: #9ece6a;
        text-style: bold;
        font-size: 24;
    }
    
    .address-item {
        padding: 1;
        margin: 1 0;
        background: #1a1b26;
        border: blank;
    }
    
    .address-item:hover {
        background: #2a2b3c;
    }
    
    .transaction-row {
        padding: 1;
    }
    
    .transaction-confirmed {
        color: #9ece6a;
    }
    
    .transaction-pending {
        color: #e0af68;
    }
    
    .transaction-failed {
        color: #f7768e;
    }
    """
    
    def __init__(self, rpc_client, app):
        super().__init__()
        self.client = rpc_client
        #self.app = app
    
    def compose(self) -> ComposeResult:
        yield Container(
            Grid(
                # Balance Overview
                self.create_balance_section(),
                
                # Quick Actions
                self.create_actions_section(),
                
                # Address Management
                self.create_addresses_section(),
                
                # Transaction History
                self.create_transactions_section(),
                
                id="wallet-container"
            )
        )
    
    def create_balance_section(self) -> Static:
        """Balance overview section"""
        return Static(
            """┌─ 💰 BALANCE OVERVIEW ──────────────┐
│                                       │
│         1,250.75 RYX                │
│         Total Balance                │
│                                       │
│  ┌─────────┬─────────┬─────────────┐ │
│  │Available│ Staked  │  Pending    │ │
│  │1000.25  │ 250.50  │   0.00      │ │
│  │  RYX    │  RYX    │   RYX       │ │
│  └─────────┴─────────┴─────────────┘ │
│                                       │
│  Primary: ryx1q8a4sm6t5r3v9x2p...   │
│                                       │
└───────────────────────────────────────┘""",
            classes="wallet-section"
        )
    
    def create_actions_section(self) -> Static:
        """Quick actions section"""
        return Static(
            """┌─ 🚀 QUICK ACTIONS ────────────────┐
│                                       │
│  ┌─────────────┬───────────────────┐ │
│  │ 📤 Send     │  Create new       │ │
│  │             │  transaction      │ │
│  ├─────────────┼───────────────────┤ │
│  │ 📥 Receive  │  Generate receive │ │
│  │             │  address          │ │
│  ├─────────────┼───────────────────┤ │
│  │ 💼 Backup   │  Encrypted wallet │ │
│  │             │  backup           │ │
│  ├─────────────┼───────────────────┤ │
│  │ 🔐 Security │  Lock/encrypt     │ │
│  │             │  wallet           │ │
│  └─────────────┴───────────────────┘ │
│                                       │
│  Press [S]end [R]eceive [B]ackup     │
│                                       │
└───────────────────────────────────────┘""",
            classes="wallet-section"
        )
    
    def create_addresses_section(self) -> Static:
        """Address management section"""
        return Static(
            """┌─ 📫 ADDRESS MANAGEMENT ───────────┐
│                                       │
│  ┌─ Address ───────┬─ Balance ─┬─ Use │
│  │ ryx1q8a4...     │ 850.25    │ 🔸   │
│  │ ryx1b7c9...     │ 150.00    │ 🔸   │
│  │ ryx1d2e5...     │   0.50    │ 🔸   │
│  │ ryx1f6g7...     │   0.00    │ 🔹   │
│  │ ryx1h8i9...     │   0.00    │ 🔹   │
│  └─────────────────┴───────────┴─────┘ │
│                                       │
│  [N]ew Address  [S]witch  [V]iew All │
│                                       │
└───────────────────────────────────────┘""",
            classes="wallet-section"
        )
    
    def create_transactions_section(self) -> Static:
        """Transaction history section"""
        return Static(
            """┌─ 📊 RECENT TRANSACTIONS ──────────┐
│                                       │
│  ┌─ Time ─┬─ Type ─┬─ Amount ─┬─ Status │
│  │ 2h ago │ Receive│ +50.0    │ ✅     │
│  │ 1d ago │ Send   │ -5.5     │ ✅     │
│  │ 3d ago │ Reward │ +1.2     │ ✅     │
│  │ 5d ago │ Send   │ -25.0    │ ✅     │
│  │ 1w ago │ Receive│ +100.0   │ ✅     │
│  └────────┴────────┴──────────┴────────┘ │
│                                       │
│  [V]iew All  [E]xport  [F]ilter       │
│                                       │
└───────────────────────────────────────┘""",
            classes="wallet-section"
        )
    
    def on_key(self, event: events.Key) -> None:
        """Handle wallet-specific keyboard shortcuts"""
        key = event.key.lower()
        
        if key == "escape" or key == "b":
            self.app.pop_screen()
        elif key == "w":
        	self.app.push_screen("wallet")
        elif key == "s":
            self.app.push_screen("send")
        elif key == "r":
            self.app.push_screen("receive")
        elif key == "n":
            self.generate_new_address()
        elif key == "1":
            self.app.push_screen("send")
        elif key == "2":
            self.app.push_screen("receive")
        elif key == "3":
            self.backup_wallet()
    
    def generate_new_address(self):
        """Generate new wallet address"""
        try:
            new_address = self.client.get_new_address()
            self.notify(f"✅ New address generated: {new_address[:16]}...")
        except Exception as e:
            self.notify(f"❌ Failed to generate address: {e}")
    
    def backup_wallet(self):
        """Backup wallet"""
        try:
            # Implementation would go here
            self.notify("💾 Wallet backup functionality")
        except Exception as e:
            self.notify(f"❌ Backup failed: {e}")