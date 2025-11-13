#!/usr/bin/env python3
"""
Network Management Screen
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Static, Button, DataTable, Sparkline
from textual.screen import Screen
from textual import events
from typing import List, Dict, Any

class NetworkScreen(Screen):
    """Network Monitoring and Management"""
    
    CSS = """
    NetworkScreen {
        align: center middle;
        background: #0f0f23;
    }
    
    #network-container {
        grid-size: 2 3;
        grid-gutter: 1 2;
        padding: 1 2;
        height: 100%;
    }
    
    .network-section {
        background: #1a1b26;
        border: round #2a2b3c;
        padding: 1 2;
    }
    
    .peer-card {
        background: #16161e;
        border: solid #2a2b3c;
        padding: 1;
        margin: 1 0;
    }
    
    .peer-connected {
        border-left: solid #9ece6a 3;
    }
    
    .peer-disconnected {
        border-left: solid #f7768e 3;
    }
    
    .peer-syncing {
        border-left: solid #e0af68 3;
    }
    
    .latency-excellent {
        color: #9ece6a;
    }
    
    .latency-good {
        color: #e0af68;
    }
    
    .latency-poor {
        color: #f7768e;
    }
    """
    
    def __init__(self, rpc_client, app):
        super().__init__()
        self.client = rpc_client
        self.app = app
    
    def compose(self) -> ComposeResult:
        yield Container(
            Grid(
                # Network Overview
                self.create_overview_section(),
                
                # Connected Peers
                self.create_peers_section(),
                
                # Performance Metrics
                self.create_performance_section(),
                
                # Network Topology
                self.create_topology_section(),
                
                # Bandwidth Usage
                self.create_bandwidth_section(),
                
                # Quick Actions
                self.create_actions_section(),
                
                id="network-container"
            )
        )
    
    def create_overview_section(self) -> Static:
        """Network overview section"""
        return Static(
            """┌─ 🌐 NETWORK OVERVIEW ────────────┐
│                                       │
│  Status:        🟢 HEALTHY           │
│  Protocol:      Rayonix v2           │
│  Network ID:    testnet-42           │
│  Client:        rayonixd/2.1.0       │
│                                       │
│  ┌─ Peers ─┬─ Sync ─┬─ Propagation ─┐ │
│  │   42    │ 99.8%  │     128ms     │ │
│  │connected│        │               │ │
│  └─────────┴────────┴───────────────┘ │
│                                       │
│  Uptime: 12 days, 4 hours            │
│  Last Restart: 2024-01-10 08:30      │
│                                       │
└───────────────────────────────────────┘""",
            classes="network-section"
        )
    
    def create_peers_section(self) -> Static:
        """Connected peers section"""
        return Static(
            """┌─ 🔗 CONNECTED PEERS ─────────────┐
│                                       │
│  ┌─ Node ─────┬─ Location ─┬─ Ping ─┐ │
│  │ ray-node-1 │ US East    │  45ms  │ │
│  │ ray-node-2 │ EU West    │  89ms  │ │
│  │ ray-node-3 │ Asia SE    │ 156ms  │ │
│  │ ray-node-4 │ US West    │  67ms  │ │
│  │ ray-node-5 │ EU North   │  92ms  │ │
│  └────────────┴────────────┴────────┘ │
│                                       │
│  Total: 42 peers connected           │
│  Incoming: 12, Outgoing: 30          │
│  Banned: 0 peers                     │
│                                       │
│  [V]iew All  [C]onnect  [B]an        │
│                                       │
└───────────────────────────────────────┘""",
            classes="network-section"
        )
    
    def create_performance_section(self) -> Static:
        """Performance metrics section"""
        return Static(
            """┌─ 📊 PERFORMANCE METRICS ─────────┐
│                                       │
│  ┌─ TPS ───┬─ Mempool ─┬─ Latency ─┐ │
│  │  45.2   │   124     │   128ms   │ │
│  │         │  transactions         │ │
│  └─────────┴───────────┴───────────┘ │
│                                       │
│  Block Propagation:                  │
│  ● Average:       128ms              │
│  ● 95th %ile:     245ms              │
│  ● Best:          45ms               │
│  ● Worst:         890ms              │
│                                       │
│  Network Hashrate: 1.2 MH/s          │
│  Difficulty:      15.4K              │
│                                       │
└───────────────────────────────────────┘""",
            classes="network-section"
        )
    
    def create_topology_section(self) -> Static:
        """Network topology section"""
        return Static(
            """┌─ 🕸️ NETWORK TOPOLOGY ────────────┐
│                                       │
│           [YOU]                       │
│              |                        │
│        ┌─────┴─────┐                  │
│        |           |                  │
│     [Peer1]     [Peer2]               │
│        |           |                  │
│    ┌───┴───┐   ┌───┴───┐              │
│    |       |   |       |              │
│ [P3]     [P4] [P5]     [P6]           │
│                                       │
│  Network Diameter: 6 hops             │
│  Average Degree:   4.2                │
│  Clustering:       0.68               │
│                                       │
│  [V]iew Full Map  [R]efresh          │
│                                       │
└───────────────────────────────────────┘""",
            classes="network-section"
        )
    
    def create_bandwidth_section(self) -> Static:
        """Bandwidth usage section"""
        return Static(
            """┌─ 📶 BANDWIDTH USAGE ─────────────┐
│                                       │
│  ┌─ Type ─────┬─ Rate ───┬─ Total ──┐ │
│  │ Download   │ 45 KB/s  │ 12.4 GB  │ │
│  │ Upload     │ 28 KB/s  │ 8.7 GB   │ │
│  │ Peak DL    │ 2.1 MB/s │ -        │ │
│  │ Peak UL    │ 1.4 MB/s │ -        │ │
│  └────────────┴──────────┴──────────┘ │
│                                       │
│  Data Transfer (24h):                │
│  ● Blocks:       1.2 GB              │
│  ● Transactions: 45 MB               │
│  ● Peers:        320 MB              │
│  ● Total:        1.6 GB              │
│                                       │
└───────────────────────────────────────┘""",
            classes="network-section"
        )
    
    def create_actions_section(self) -> Static:
        """Quick actions section"""
        return Static(
            """┌─ 🚀 QUICK ACTIONS ────────────────┐
│                                       │
│  ┌─────────────┬───────────────────┐ │
│  │ 🔍 Discover │  Find new peers   │ │
│  │             │  automatically    │ │
│  ├─────────────┼───────────────────┤ │
│  │ ➕ Connect   │  Connect to       │ │
│  │             │  specific peer    │ │
│  ├─────────────┼───────────────────┤ │
│  │ 🚫 Ban      │  Ban malicious    │ │
│  │             │  peer             │ │
│  ├─────────────┼───────────────────┤ │
│  │ 📊 Metrics  │  Detailed network │ │
│  │             │  statistics       │ │
│  └─────────────┴───────────────────┘ │
│                                       │
│  [D]iscover [C]onnect [B]an [M]etrics│
│                                       │
└───────────────────────────────────────┘""",
            classes="network-section"
        )
    
    def on_key(self, event: events.Key) -> None:
        """Handle network keyboard shortcuts"""
        key = event.key.lower()
        
        if key == "escape" or key == "b":
            self.app.pop_screen()
        elif key == "d":
            self.discover_peers()
        elif key == "c":
            self.connect_peer()
        elif key == "b":
            self.ban_peer()
        elif key == "m":
            self.show_metrics()
        elif key == "1":
            self.discover_peers()
        elif key == "2":
            self.connect_peer()
    
    def discover_peers(self):
        """Discover new peers"""
        try:
            self.notify("🔍 Discovering new peers...")
        except Exception as e:
            self.notify(f"❌ Peer discovery failed: {e}")
    
    def connect_peer(self):
        """Connect to specific peer"""
        try:
            self.notify("➕ Connect to peer interface")
        except Exception as e:
            self.notify(f"❌ Connection failed: {e}")
    
    def ban_peer(self):
        """Ban malicious peer"""
        try:
            self.notify("🚫 Ban peer interface")
        except Exception as e:
            self.notify(f"❌ Ban failed: {e}")
    
    def show_metrics(self):
        """Show detailed metrics"""
        try:
            self.notify("📊 Detailed network metrics interface")
        except Exception as e:
            self.notify(f"❌ Failed to load metrics: {e}")