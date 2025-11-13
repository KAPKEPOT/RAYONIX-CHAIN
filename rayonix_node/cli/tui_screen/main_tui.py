#!/usr/bin/env python3
"""
Main TUI Application - Complete with all screens
"""

import asyncio
from textual.app import App
from textual.widgets import Header, Footer

# Import all screens
from rayonix_node.cli.tui_screen.dashboard import DashboardScreen
from rayonix_node.cli.tui_screenwallet_screen import WalletScreen
from rayonix_node.cli.tui_screenstaking_screen import StakingScreen
from rayonix_node.cli.tui_screenapi_screen import ApiScreen
from rayonix_node.cli.tui_screencontracts_screen import ContractsScreen
from rayonix_node.cli.tui_screennetwork_screen import NetworkScreen
from rayonix_node.cli.tui_screenvalidators_screen import ValidatorsScreen

class RayonixTUI(App):
    """Complete Rayonix TUI Application"""
    
    CSS = """
    App {
        background: #0f0f23;
    }
    
    Header {
        background: #1a1b26;
        color: #7aa2f7;
        text-style: bold;
        border: solid #2a2b3c;
    }
    
    Footer {
        background: #1a1b26;
        color: #565f89;
        border-top: solid #2a2b3c;
    }
    
    .screen-title {
        color: #7aa2f7;
        text-style: bold;
        text-align: center;
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "show_dashboard", "Dashboard"),
        ("w", "show_wallet", "Wallet"),
        ("s", "show_staking", "Staking"),
        ("a", "show_api", "API"),
        ("c", "show_contracts", "Contracts"),
        ("n", "show_network", "Network"),
        ("v", "show_validators", "Validators"),
        ("f1", "show_help", "Help"),
    ]
    
    def __init__(self, rpc_client, data_dir: str):
        super().__init__()
        self.client = rpc_client
        self.data_dir = data_dir
        
    def on_mount(self) -> None:
        """Initialize the application with all screens"""
        # Register all screens
        self.install_screen(DashboardScreen(self.client, self), name="dashboard")
        self.install_screen(WalletScreen(self.client, self), name="wallet")
        self.install_screen(StakingScreen(self.client, self), name="staking")
        self.install_screen(ApiScreen(self.client, self), name="api")
        self.install_screen(ContractsScreen(self.client, self), name="contracts")
        self.install_screen(NetworkScreen(self.client, self), name="network")
        self.install_screen(ValidatorsScreen(self.client, self), name="validators")
        
        # Start with dashboard
        self.push_screen("dashboard")
    
    def create_header(self) -> Header:
        """Create application header"""
        return Header()
    
    def create_footer(self) -> Footer:
        """Create application footer"""
        return Footer()
    
    # Navigation actions
    def action_show_dashboard(self) -> None:
        self.push_screen("dashboard")
    
    def action_show_wallet(self) -> None:
        self.push_screen("wallet")
    
    def action_show_staking(self) -> None:
        self.push_screen("staking")
    
    def action_show_api(self) -> None:
        self.push_screen("api")
    
    def action_show_contracts(self) -> None:
        self.push_screen("contracts")
    
    def action_show_network(self) -> None:
        self.push_screen("network")
    
    def action_show_validators(self) -> None:
        self.push_screen("validators")
    
    def action_show_help(self) -> None:
        """Show help screen with all shortcuts"""
        help_text = """
🌐 RAYONIX TUI - KEYBOARD SHORTCUTS

NAVIGATION:
  D - Dashboard      W - Wallet        S - Staking
  A - API Management C - Contracts     N - Network  
  V - Validators    F1 - Help         Q - Quit

QUICK ACTIONS (Dashboard):
  1 - Send Funds     2 - Receive      3 - Stake
  4 - Deploy Contract5 - Create API Key

GLOBAL:
  ESC/B - Back to previous screen
        """
        self.notify(help_text, title="Help", severity="information")

def run_modern_tui(rpc_client, data_dir: str):
    """Run the complete modern TUI interface"""
    try:
        app = RayonixTUI(rpc_client, data_dir)
        app.run()
    except Exception as e:
        print(f"❌ Modern TUI failed: {e}")
        print("🔄 Falling back to classic interface...")
        # Fallback implementation would go here