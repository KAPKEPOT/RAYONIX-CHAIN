#!/usr/bin/env python3
"""
RAYONIX Modern TUI Interface
Seamlessly integrates with existing codebase
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, DataTable, Input, 
    Select, Label, Markdown, ProgressBar
)
from textual.reactive import reactive
from textual.screen import Screen
from textual import events
import json

# Import existing components
from rayonix_node.cli.interactive import RayonixInteractiveCLI
from rayonix_node.cli.command_handler import CommandHandler
from rayonix_cli import RayonixRPCClient

class RayonixTUI(App):
    """Modern Rayonix Terminal User Interface"""
    
    CSS = """
    Screen {
        background: #0f1116;
    }
    
    #dashboard {
        layout: grid;
        grid-size: 2 3;
        grid-gutter: 1;
        padding: 1;
        height: 100%;
    }
    
    .card {
        background: #1a1d26;
        border: solid #2d313e;
        border-title-color: #6c727f;
        padding: 1;
        height: 100%;
    }
    
    .card-title {
        color: #6c727f;
        text-style: bold;
    }
    
    .status-online {
        color: #27d796;
    }
    
    .status-offline {
        color: #e5484d;
    }
    
    .status-syncing {
        color: #ffd33d;
    }
    
    .balance-positive {
        color: #27d796;
    }
    
    .balance-negative {
        color: #e5484d;
    }
    
    .button-primary {
        background: #3e63dd;
    }
    
    .button-success {
        background: #27d796;
    }
    
    DataTable {
        background: #1a1d26;
    }
    
    #main-footer {
        background: #1a1d26;
        color: #6c727f;
    }
    """
    
    # Reactive state
    node_status = reactive("connecting")
    block_height = reactive(0)
    wallet_balance = reactive(0.0)
    connected_peers = reactive(0)
    sync_progress = reactive(0.0)
    
    def __init__(self, rpc_client: RayonixRPCClient):
        super().__init__()
        self.client = rpc_client
        self.command_handler = CommandHandler(rpc_client)
        self.current_screen = "dashboard"
        
    def compose(self) -> ComposeResult:
        """Create the main UI layout"""
        yield Header()
        yield Container(
            self.create_dashboard(),
            id="main-container"
        )
        yield Footer(id="main-footer")
    
    def create_dashboard(self) -> Container:
        """Create the main dashboard"""
        return Container(
            Horizontal(
                # Left sidebar - Navigation
                Vertical(
                    Static("🌐 RAYONIX", classes="card-title"),
                    Button("📊 Dashboard", variant="primary", id="btn-dashboard"),
                    Button("💰 Wallet", variant="default", id="btn-wallet"),
                    Button("⚡ Staking", variant="default", id="btn-staking"),
                    Button("🔐 API", variant="default", id="btn-api"),
                    Button("🤖 Contracts", variant="default", id="btn-contracts"),
                    Button("🌐 Network", variant="default", id="btn-network"),
                    classes="card",
                    id="navigation"
                ),
                
                # Main content area
                Vertical(
                    self.create_status_card(),
                    self.create_wallet_card(),
                    self.create_staking_card(),
                    self.create_network_card(),
                    id="main-content"
                ),
            ),
            id="dashboard"
        )
    
    def create_status_card(self) -> Static:
        """Create node status card"""
        return Static(
            """╭─ 📊 NODE STATUS ─────────────────────────────────╮
│                                                          │
│  Status:    [status] Connecting...                      │
│  Block:     #0                                          │
│  Peers:     0 connected                                 │
│  Sync:      0%                                          │
│  Version:   v2.1.0                                      │
│                                                          │
╰──────────────────────────────────────────────────────────╯""",
            classes="card",
            id="status-card"
        )
    
    def create_wallet_card(self) -> Static:
        """Create wallet overview card"""
        return Static(
            """╭─ 💰 WALLET ─────────────────────────────────────╮
│                                                          │
│  Balance:   0.00 RYX                                    │
│  Status:    Not loaded                                  │
│  Addresses: 0                                           │
│                                                          │
│  [S]end  [R]eceive  [H]istory                           │
│                                                          │
╰──────────────────────────────────────────────────────────╯""",
            classes="card",
            id="wallet-card"
        )
    
    def create_staking_card(self) -> Static:
        """Create staking overview card"""
        return Static(
            """╭─ ⚡ STAKING ────────────────────────────────────╮
│                                                          │
│  Status:    Inactive                                    │
│  Staked:    0.00 RYX                                    │
│  Rewards:   0.00 RYX                                    │
│  APR:       0%                                          │
│                                                          │
│  [D]elegate  [C]laim  [V]alidators                      │
│                                                          │
╰──────────────────────────────────────────────────────────╯""",
            classes="card",
            id="staking-card"
        )
    
    def create_network_card(self) -> Static:
        """Create network overview card"""
        return Static(
            """╭─ 🌐 NETWORK ───────────────────────────────────╮
│                                                          │
│  TPS:       0.0                                         │
│  Mempool:   0 transactions                              │
│  Latency:   0ms                                         │
│  Hashrate:  0 H/s                                       │
│                                                          │
│  [P]eers  [M]etrics  [T]opology                         │
│                                                          │
╰──────────────────────────────────────────────────────────╯""",
            classes="card",
            id="network-card"
        )
    
    async def on_mount(self) -> None:
        """Initialize the application"""
        self.set_interval(5, self.update_data)  # Update every 5 seconds
        await self.update_data()  # Initial data load
    
    async def update_data(self) -> None:
        """Update all dashboard data"""
        try:
            # Get node status
            status = await self.get_node_status()
            self.node_status = status.get('status', 'unknown')
            self.block_height = status.get('block_height', 0)
            self.connected_peers = status.get('peers_connected', 0)
            self.sync_progress = status.get('sync_progress', 0)
            
            # Get wallet balance if wallet is loaded
            try:
                balance_info = self.client.get_wallet_detailed_balance()
                self.wallet_balance = balance_info.get('total', 0.0)
            except:
                self.wallet_balance = 0.0
            
            # Update UI
            await self.update_status_card()
            await self.update_wallet_card()
            await self.update_staking_card()
            await self.update_network_card()
            
        except Exception as e:
            self.log(f"Error updating data: {e}")
    
    async def get_node_status(self) -> Dict[str, Any]:
        """Get node status from RPC client"""
        try:
            return self.client.get_node_status()
        except Exception as e:
            return {
                'status': 'offline',
                'block_height': 0,
                'peers_connected': 0,
                'sync_progress': 0,
                'error': str(e)
            }
    
    async def update_status_card(self) -> None:
        """Update status card with current data"""
        status_icon = {
            'online': '🟢 Online',
            'offline': '🔴 Offline', 
            'syncing': '🟡 Syncing',
            'connecting': '🟠 Connecting',
            'unknown': '⚪ Unknown'
        }.get(self.node_status, '⚪ Unknown')
        
        status_card = self.query_one("#status-card", Static)
        status_content = f"""╭─ 📊 NODE STATUS ─────────────────────────────────╮
│                                                          │
│  Status:    {status_icon:<30} │
│  Block:     #{self.block_height:<28} │
│  Peers:     {self.connected_peers} connected{' ' * 20}│
│  Sync:      {self.sync_progress}%{' ' * 27}│
│  Version:   v2.1.0{' ' * 28}│
│                                                          │
╰──────────────────────────────────────────────────────────╯"""
        
        status_card.update(status_content)
    
    async def update_wallet_card(self) -> None:
        """Update wallet card with current data"""
        try:
            wallet_info = self.client.get_wallet_info()
            address_count = len(wallet_info.get('addresses', []))
            wallet_status = "🔓 Loaded" if address_count > 0 else "🔒 Not loaded"
        except:
            address_count = 0
            wallet_status = "🔒 Not loaded"
        
        wallet_card = self.query_one("#wallet-card", Static)
        wallet_content = f"""╭─ 💰 WALLET ─────────────────────────────────────╮
│                                                          │
│  Balance:   {self.wallet_balance:>8.2f} RYX{' ' * 18}│
│  Status:    {wallet_status:<28} │
│  Addresses: {address_count}{' ' * 28}│
│                                                          │
│  [S]end  [R]eceive  [H]istory{' ' * 18}│
│                                                          │
╰──────────────────────────────────────────────────────────╯"""
        
        wallet_card.update(wallet_content)
    
    async def update_staking_card(self) -> None:
        """Update staking card with current data"""
        try:
            staking_info = self.client.get_staking_info()
            staked_amount = staking_info.get('total_staked', 0.0)
            rewards = staking_info.get('expected_rewards', 0.0)
            staking_status = "🟢 Active" if staking_info.get('enabled', False) else "⚪ Inactive"
            apr = staking_info.get('apr', 0.0)
        except:
            staked_amount = 0.0
            rewards = 0.0
            staking_status = "⚪ Inactive"
            apr = 0.0
        
        staking_card = self.query_one("#staking-card", Static)
        staking_content = f"""╭─ ⚡ STAKING ────────────────────────────────────╮
│                                                          │
│  Status:    {staking_status:<28} │
│  Staked:    {staked_amount:>8.2f} RYX{' ' * 18}│
│  Rewards:   {rewards:>8.2f} RYX{' ' * 18}│
│  APR:       {apr:>5.1f}%{' ' * 25}│
│                                                          │
│  [D]elegate  [C]laim  [V]alidators{' ' * 16}│
│                                                          │
╰──────────────────────────────────────────────────────────╯"""
        
        staking_card.update(staking_content)
    
    async def update_network_card(self) -> None:
        """Update network card with current data"""
        try:
            network_stats = self.client.get_network_stats()
            tps = network_stats.get('transactions_per_second', 0.0)
            mempool_size = network_stats.get('mempool_size', 0)
            latency = network_stats.get('average_latency', 0)
            hashrate = network_stats.get('network_hashrate', 0)
        except:
            tps = 0.0
            mempool_size = 0
            latency = 0
            hashrate = 0
        
        network_card = self.query_one("#network-card", Static)
        network_content = f"""╭─ 🌐 NETWORK ───────────────────────────────────╮
│                                                          │
│  TPS:       {tps:>5.1f}{' ' * 26}│
│  Mempool:   {mempool_size} transactions{' ' * 16}│
│  Latency:   {latency}ms{' ' * 26}│
│  Hashrate:  {hashrate} H/s{' ' * 22}│
│                                                          │
│  [P]eers  [M]etrics  [T]opology{' ' * 17}│
│                                                          │
╰──────────────────────────────────────────────────────────╯"""
        
        network_card.update(network_content)
    
    def on_key(self, event: events.Key) -> None:
        """Handle global keyboard shortcuts"""
        key = event.key.lower()
        
        if key == "q":
            self.exit()
        elif key == "w":
            self.push_screen("wallet")
        elif key == "s":
            self.push_screen("staking")
        elif key == "a":
            self.push_screen("api")
        elif key == "c":
            self.push_screen("contracts")
        elif key == "n":
            self.push_screen("network")
        elif key == "d":
            self.app.pop_screen()  # Back to dashboard
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        button_id = event.button.id
        
        if button_id == "btn-dashboard":
            pass  # Already on dashboard
        elif button_id == "btn-wallet":
            self.push_screen("wallet")
        elif button_id == "btn-staking":
            self.push_screen("staking")
        elif button_id == "btn-api":
            self.push_screen("api")
        elif button_id == "btn-contracts":
            self.push_screen("contracts")
        elif button_id == "btn-network":
            self.push_screen("network")

class WalletScreen(Screen):
    """Wallet Management Screen"""
    
    CSS = """
    WalletScreen {
        background: #0f1116;
    }
    
    #wallet-container {
        padding: 1;
    }
    
    .balance-display {
        background: #1a1d26;
        border: solid #27d796;
        padding: 2;
        text-align: center;
    }
    
    .transaction-table {
        height: 20;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("💰 WALLET MANAGER", classes="card-title"),
            self.create_balance_display(),
            self.create_quick_actions(),
            self.create_transaction_history(),
            id="wallet-container"
        )
        yield Footer()
    
    def create_balance_display(self) -> Static:
        return Static(
            """╭─ BALANCE OVERVIEW ──────────────────────────────╮
│                                                          │
│                1,250.75 RYX                            │
│                Available                                │
│                                                          │
│    Primary: ryx1q8a4sm6t5r3v9x2p4q6r8s0t1u3v5w7x9y0z1  │
│                                                          │
╰──────────────────────────────────────────────────────────╯""",
            classes="balance-display"
        )
    
    def create_quick_actions(self) -> Horizontal:
        return Horizontal(
            Button("📤 Send", variant="primary", id="send-btn"),
            Button("📥 Receive", variant="success", id="receive-btn"), 
            Button("📊 History", variant="default", id="history-btn"),
            Button("🔐 Backup", variant="default", id="backup-btn"),
            Button("← Back", variant="default", id="back-btn"),
        )
    
    def create_transaction_history(self) -> DataTable:
        table = DataTable()
        table.add_columns("Time", "Description", "Amount", "Status")
        table.add_rows([
            ["2h ago", "Received Payment", "+50.0 RYX", "✅ Confirmed"],
            ["1d ago", "Coffee Shop", "-5.5 RYX", "✅ Confirmed"],
            ["3d ago", "Staking Reward", "+1.2 RYX", "✅ Confirmed"],
        ])
        return table
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "send-btn":
            self.app.push_screen(SendScreen())

class SendScreen(Screen):
    """Send Funds Screen"""
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("📤 SEND FUNDS", classes="card-title"),
            Input(placeholder="To address...", id="to-address"),
            Input(placeholder="Amount (RYX)...", id="amount"),
            Static("Fee: 0.001 RYX", id="fee-display"),
            Horizontal(
                Button("Send", variant="primary", id="send-confirm"),
                Button("Cancel", variant="default", id="send-cancel"),
            ),
            id="send-container"
        )
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-cancel":
            self.app.pop_screen()

def run_modern_tui(rpc_client: RayonixRPCClient, data_dir: str):
    """Run the modern TUI interface"""
    try:
        app = RayonixTUI(rpc_client)
        app.run()
    except Exception as e:
        print(f"Error starting modern TUI: {e}")
        # Fallback to traditional interactive mode
        from rayonix_node.cli.interactive import run_interactive_mode
        run_interactive_mode(rpc_client, data_dir)