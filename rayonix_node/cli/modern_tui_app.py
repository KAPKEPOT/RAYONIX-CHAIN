"""
Modern TUI Application - Separate file to avoid circular imports
"""

import asyncio
from typing import Dict, Any
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Button

class RayonixTUI(App):
    """Modern Rayonix Terminal User Interface"""
    
    CSS = """
    Screen {
        background: #0f1116;
    }
    
    .dashboard {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 1;
    }
    
    .card {
        background: #1a1d26;
        border: solid #2d313e;
        padding: 1;
    }
    
    .card-title {
        color: #6c727f;
        text-style: bold;
    }
    """
    
    def __init__(self, rpc_client, data_dir: str):
        super().__init__()
        self.client = rpc_client
        self.data_dir = data_dir
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Horizontal(
                    self.create_status_card(),
                    self.create_wallet_card(),
                ),
                Horizontal(
                    self.create_staking_card(),
                    self.create_network_card(),
                ),
                id="dashboard"
            )
        )
        yield Footer()
    
    def create_status_card(self) -> Static:
        return Static(
            """╭─ 📊 NODE STATUS ─────────────────────────╮
│  🟢 Online  │  🔄 Synced (99.8%)         │
│  Block: #1,248,752  │  Peers: 42        │
╰──────────────────────────────────────────╯""",
            classes="card"
        )
    
    def create_wallet_card(self) -> Static:
        return Static(
            """╭─ 💰 WALLET ──────────────────────────────╮
│  Balance: 1,250.75 RYX                 │
│  Status: 🔓 Loaded                     │
│  Addresses: 15                         │
╰──────────────────────────────────────────╯""",
            classes="card"
        )
    
    def create_staking_card(self) -> Static:
        return Static(
            """╭─ ⚡ STAKING ─────────────────────────────╮
│  Status: 🟢 Active                    │
│  Staked: 250.50 RYX                   │
│  Rewards: 1.75 RYX                    │
╰──────────────────────────────────────────╯""",
            classes="card"
        )
    
    def create_network_card(self) -> Static:
        return Static(
            """╭─ 🌐 NETWORK ────────────────────────────╮
│  TPS: 45.2                          │
│  Mempool: 124 tx                    │
│  Hashrate: 1.2 MH/s                 │
╰──────────────────────────────────────────╯""",
            classes="card"
        )
    
    def on_key(self, event):
        if event.key == "q":
            self.exit()
        elif event.key == "w":
            self.push_screen("wallet")
    
    def on_button_pressed(self, event: Button.Pressed):
        self.exit(message="Navigation coming soon!")