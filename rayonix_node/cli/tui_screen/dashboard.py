#!/usr/bin/env python3
"""
RAYONIX Modern Dashboard - Main Screen (FIXED)
"""

import time
from typing import Dict, Any, List
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Static, Button, DataTable, ProgressBar
from textual.reactive import reactive
from textual.screen import Screen
from textual import events

class DashboardScreen(Screen):
    """Main Dashboard with comprehensive overview - FIXED VERSION"""
    
    CSS = """
    DashboardScreen {
        align: center middle;
        background: #0f0f23;
    }
    
    #dashboard-grid {
        grid-size: 3 3;
        grid-gutter: 1 2;
        padding: 1 2;
        height: 100%;
    }
    
    .dashboard-card {
        background: #1a1b26;
        border: round #2a2b3c;
        padding: 1 2;
    }
    
    .card-title {
        color: #7aa2f7;
        text-style: bold;
        margin-bottom: 1;
    }
    """
    
    # Reactive state
    node_status = reactive("connecting")
    block_height = reactive(0)
    wallet_balance = reactive(0.0)
    connected_peers = reactive(0)
    sync_progress = reactive(0)
    network_activity = reactive([0] * 20)
    
    def __init__(self, rpc_client):
        super().__init__()
        self.client = rpc_client
        self.last_update = 0
    
    def compose(self) -> ComposeResult:
        """Create the dashboard layout"""
        yield Container(
            Grid(
                # Row 1: Status Cards
                self.create_status_card(),
                self.create_wallet_card(),
                self.create_staking_card(),
                
                # Row 2: Network & Performance
                self.create_network_card(),
                self.create_performance_card(),
                self.create_quick_actions_card(),
                
                # Row 3: Recent Activity & System Info
                self.create_activity_card(),
                self.create_system_card(),
                self.create_validators_card(),
                
                id="dashboard-grid"
            )
        )
    
    def create_status_card(self) -> Static:
        """Node status card"""
        return Static(
            """┌─ 🏠 NODE STATUS ──────────────────┐
│                                       │
│  ● STATUS:    CONNECTING             │
│  ● BLOCK:     #0                     │
│  ● PEERS:     0 connected            │
│  ● SYNC:      [                    ] │
│  ● UPTIME:    0d 0h 0m              │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    def create_wallet_card(self) -> Static:
        """Wallet overview card"""
        return Static(
            """┌─ 💰 WALLET ───────────────────────┐
│                                       │
│           0.00 RYX                   │
│           Total Balance              │
│                                       │
│  ● Available:  0.00 RYX              │
│  ● Staked:     0.00 RYX              │
│  ● Pending:    0.00 RYX              │
│  ● Addresses:  0                     │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    def create_staking_card(self) -> Static:
        """Staking overview card"""
        return Static(
            """┌─ ⚡ STAKING ──────────────────────┐
│                                       │
│  ● Status:    INACTIVE               │
│  ● Staked:    0.00 RYX               │
│  ● Rewards:   0.00 RYX               │
│  ● APR:       0.0%                   │
│  ● Rank:      -/-                    │
│                                       │
│  [Delegate] [Claim] [Validators]     │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    def create_network_card(self) -> Static:
        """Network activity card"""
        return Static(
            """┌─ 🌐 NETWORK ──────────────────────┐
│                                       │
│  ┌─────────────────────────────────┐ │
│  │                                 │ │
│  │                                 │ │
│  └─────────────────────────────────┘ │
│  ● TPS:        0.0                   │
│  ● Mempool:    0 transactions        │
│  ● Latency:    0ms                   │
│  ● Hashrate:   0 H/s                 │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    def create_performance_card(self) -> Static:
        """Performance metrics card"""
        return Static(
            """┌─ 📊 PERFORMANCE ─────────────────┐
│                                       │
│  CPU:    [          ] 0%             │
│  Memory: [          ] 0%             │
│  Disk:   [          ] 0%             │
│                                       │
│  ● Block Time:   0.0s                │
│  ● Propagation:  0ms                 │
│  ● Cache Hit:    0%                  │
│  ● Requests:     0/sec               │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    def create_quick_actions_card(self) -> Static:
        """Quick actions card"""
        return Static(
            """┌─ 🚀 QUICK ACTIONS ────────────────┐
│                                       │
│  ┌───────┐ ┌───────┐ ┌───────┐       │
│  │  Send │ │Receive│ │Stake  │       │
│  └───────┘ └───────┘ └───────┘       │
│                                       │
│  ┌───────┐ ┌───────┐ ┌───────┐       │
│  │Contracts││ API   │ │Backup │       │
│  └───────┘ └───────┘ └───────┘       │
│                                       │
│  Press [H] for help                  │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    def create_activity_card(self) -> Static:
        """Recent activity card"""
        return Static(
            """┌─ 📈 RECENT ACTIVITY ──────────────┐
│                                       │
│  No recent activity                  │
│                                       │
│  ┌─ Time ─┬─ Description ─┬─ Amount ─┐ │
│  │        │               │          │ │
│  │        │               │          │ │
│  │        │               │          │ │
│  └────────┴───────────────┴──────────┘ │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    def create_system_card(self) -> Static:
        """System information card"""
        return Static(
            """┌─ 🔧 SYSTEM INFO ─────────────────┐
│                                       │
│  ● Version:    v2.1.0                │
│  ● Network:    testnet               │
│  ● Consensus:  PoS                   │
│  ● Database:   Operational           │
│  ● API:        Disabled              │
│  ● Auto-Update:Available             │
│                                       │
│  Last Backup: Never                  │
│  Security Scan: Required             │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    def create_validators_card(self) -> Static:
        """Validators overview card"""
        return Static(
            """┌─ 👑 VALIDATORS ──────────────────┐
│                                       │
│  Total Validators: 0                 │
│  Active:           0                 │
│  Inactive:         0                 │
│  Average Fee:      0%                │
│                                       │
│  Top Validators:                     │
│  1. -                               │
│  2. -                               │
│  3. -                               │
│                                       │
└───────────────────────────────────────┘""",
            classes="dashboard-card"
        )
    
    async def on_mount(self) -> None:
        """Initialize dashboard"""
        self.set_interval(2, self.update_dashboard)  # Update every 2 seconds
        await self.update_dashboard()
    
    async def update_dashboard(self) -> None:
        """Update all dashboard components"""
        try:
            # Get node status
            status = await self.get_node_status()
            self.node_status = status.get('status', 'unknown')
            self.block_height = status.get('block_height', 0)
            self.connected_peers = status.get('peers_connected', 0)
            self.sync_progress = status.get('sync_progress', 0)
            
            # Update network activity
            self.update_network_activity()
            
            # Update all cards
            await self.update_status_card()
            await self.update_wallet_card()
            await self.update_staking_card()
            await self.update_network_card()
            
        except Exception as e:
            self.log(f"Dashboard update error: {e}")
    
    def update_network_activity(self):
        """Simulate network activity"""
        import random
        if len(self.network_activity) >= 20:
            self.network_activity.pop(0)
        self.network_activity.append(random.randint(0, 100))
    
    async def update_status_card(self):
        """Update status card with live data"""
        status_icons = {
            'online': '🟢 ONLINE',
            'offline': '🔴 OFFLINE',
            'syncing': '🟡 SYNCING', 
            'connecting': '🟠 CONNECTING'
        }
        
        status_text = status_icons.get(self.node_status, '⚪ UNKNOWN')
        
        # Create progress bar
        progress_bar = "[" + "█" * int(self.sync_progress / 5) + " " * (20 - int(self.sync_progress / 5)) + "]"
        
        status_content = f"""┌─ 🏠 NODE STATUS ──────────────────┐
│                                       │
│  ● STATUS:    {status_text:<20} │
│  ● BLOCK:     #{self.block_height:<19} │
│  ● PEERS:     {self.connected_peers} connected{' ' * (11 - len(str(self.connected_peers)))}│
│  ● SYNC:      {progress_bar} │
│  ● UPTIME:    0d 0h 0m{' ' * 12}│
│                                       │
└───────────────────────────────────────┘"""
        
        status_card = self.query_one(".dashboard-card")  # First card
        status_card.update(status_content)
    
    async def update_wallet_card(self):
        """Update wallet card with live data"""
        try:
            wallet_info = self.client.get_wallet_info()
            balance_info = self.client.get_wallet_detailed_balance()
            addresses = self.client.get_wallet_addresses()
            
            total_balance = balance_info.get('total', 0.0)
            available = balance_info.get('available', 0.0)
            staked = balance_info.get('staked', 0.0)
            pending = balance_info.get('pending', 0.0)
            address_count = len(addresses)
        except:
            total_balance = available = staked = pending = 0.0
            address_count = 0
        
        wallet_content = f"""┌─ 💰 WALLET ───────────────────────┐
│                                       │
│           {total_balance:>8.2f} RYX{' ' * 11}│
│           Total Balance{' ' * 13}│
│                                       │
│  ● Available:  {available:>8.2f} RYX{' ' * 5}│
│  ● Staked:     {staked:>8.2f} RYX{' ' * 5}│
│  ● Pending:    {pending:>8.2f} RYX{' ' * 5}│
│  ● Addresses:  {address_count}{' ' * (16 - len(str(address_count)))}│
│                                       │
└───────────────────────────────────────┘"""
        
        # Update the second card (wallet card)
        wallet_cards = self.query(".dashboard-card")
        if len(wallet_cards) > 1:
            wallet_cards[1].update(wallet_content)
    
    async def update_staking_card(self):
        """Update staking card with live data"""
        try:
            staking_info = self.client.get_staking_info()
            staked = staking_info.get('total_staked', 0.0)
            rewards = staking_info.get('expected_rewards', 0.0)
            enabled = staking_info.get('enabled', False)
            status = "🟢 ACTIVE" if enabled else "⚪ INACTIVE"
        except:
            staked = rewards = 0.0
            status = "⚪ INACTIVE"
        
        staking_content = f"""┌─ ⚡ STAKING ──────────────────────┐
│                                       │
│  ● Status:    {status:<20} │
│  ● Staked:    {staked:>8.2f} RYX{' ' * 5}│
│  ● Rewards:   {rewards:>8.2f} RYX{' ' * 5}│
│  ● APR:       0.0%{' ' * 17}│
│  ● Rank:      -/-{' ' * 19}│
│                                       │
│  [Delegate] [Claim] [Validators]     │
│                                       │
└───────────────────────────────────────┘"""
        
        # Update the third card (staking card)
        staking_cards = self.query(".dashboard-card")
        if len(staking_cards) > 2:
            staking_cards[2].update(staking_content)
    
    async def update_network_card(self):
        """Update network card with live data"""
        try:
            network_stats = self.client.get_network_stats()
            tps = network_stats.get('transactions_per_second', 0.0)
            mempool = network_stats.get('mempool_size', 0)
            latency = network_stats.get('average_latency', 0)
            hashrate = network_stats.get('network_hashrate', 0)
        except:
            tps = 0.0
            mempool = 0
            latency = 0
            hashrate = 0
        
        # Create simple sparkline
        sparkline = ""
        for value in self.network_activity[-10:]:
            height = int(value / 20)  # Scale to 0-5
            sparkline += "▁▂▃▄▅▆▇"[height] if height < 8 else "█"
        
        network_content = f"""┌─ 🌐 NETWORK ──────────────────────┐
│                                       │
│  ┌{sparkline:─<10}┐{' ' * 15}│
│  │{'Network Activity':^24}│ │
│  └──────────────────────────┘ │
│  ● TPS:        {tps:>5.1f}{' ' * 14}│
│  ● Mempool:    {mempool} transactions{' ' * (6 - len(str(mempool)))}│
│  ● Latency:    {latency}ms{' ' * (16 - len(str(latency)))}│
│  ● Hashrate:   {hashrate} H/s{' ' * (11 - len(str(hashrate)))}│
│                                       │
└───────────────────────────────────────┘"""
        
        # Update the fourth card (network card)
        network_cards = self.query(".dashboard-card")
        if len(network_cards) > 3:
            network_cards[3].update(network_content)
    
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
    
    def on_key(self, event: events.Key) -> None:
        """Handle keyboard navigation"""
        key = event.key.lower()
        
        if key == "q":
            self.app.exit()
        elif key == "w":
            self.app.push_screen("wallet")
        elif key == "s":
            self.app.push_screen("staking")
        elif key == "a":
            self.app.push_screen("api")
        elif key == "c":
            self.app.push_screen("contracts")
        elif key == "n":
            self.app.push_screen("network")
        elif key == "v":
            self.app.push_screen("validators")
        elif key == "h":
            self.show_help()
        elif key == "1":
            self.app.push_screen("send")
        elif key == "2":
            self.app.push_screen("receive")
        elif key == "3":
            self.app.push_screen("stake")
    
    def show_help(self):
        """Show help information"""
        help_text = """
🌐 RAYONIX DASHBOARD - KEYBOARD SHORTCUTS

NAVIGATION:
  W - Wallet        S - Staking      A - API
  C - Contracts     N - Network      V - Validators
  Q - Quit          H - Help

QUICK ACTIONS:
  1 - Send Funds    2 - Receive     3 - Stake Funds

DASHBOARD FEATURES:
  • Real-time node status
  • Live wallet balance
  • Staking overview
  • Network metrics
  • Recent activity
        """
        self.app.notify(help_text, title="Dashboard Help", severity="information")