#!/usr/bin/env python3
"""
API Management Screen
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Static, Button, DataTable, Input, Switch
from textual.screen import Screen
from textual import events
from typing import List, Dict, Any
from datetime import datetime, timedelta

class ApiScreen(Screen):
    """API Key and Security Management"""
    
    CSS = """
    ApiScreen {
        align: center middle;
        background: #0f0f23;
    }
    
    #api-container {
        grid-size: 2 3;
        grid-gutter: 1 2;
        padding: 1 2;
        height: 100%;
    }
    
    .api-section {
        background: #1a1b26;
        border: round #2a2b3c;
        padding: 1 2;
    }
    
    .api-key-item {
        background: #16161e;
        border: solid #2a2b3c;
        padding: 1;
        margin: 1 0;
    }
    
    .api-key-item:hover {
        background: #2a2b3c;
        border: solid #7aa2f7;
    }
    
    .security-critical {
        color: #f7768e;
        text-style: bold;
    }
    
    .security-warning {
        color: #e0af68;
    }
    
    .security-ok {
        color: #9ece6a;
    }
    
    .rate-limit-low {
        color: #9ece6a;
    }
    
    .rate-limit-medium {
        color: #e0af68;
    }
    
    .rate-limit-high {
        color: #f7768e;
    }
    """
    
    def __init__(self, rpc_client):
        super().__init__()
        self.client = rpc_client
       # self.app = app
    
    def compose(self) -> ComposeResult:
        yield Container(
            Grid(
                # API Server Status
                self.create_status_section(),
                
                # Active API Keys
                self.create_keys_section(),
                
                # Security Settings
                self.create_security_section(),
                
                # Usage Statistics
                self.create_usage_section(),
                
                # Rate Limiting
                self.create_limits_section(),
                
                # Quick Actions
                self.create_actions_section(),
                
                id="api-container"
            )
        )
    
    def create_status_section(self) -> Static:
        """API server status section"""
        return Static(
            """┌─ 🔐 API SERVER STATUS ────────────┐
│                                       │
│  Status:        🟢 RUNNING           │
│  Port:          52557                │
│  Protocol:      HTTP/1.1             │
│  Start Time:    2 days ago           │
│                                       │
│  ┌─ Requests ─┬─ Success ─┬─ Avg RT ─┐ │
│  │    142     │   97.2%   │   45ms   │ │
│  │  (24h)     │           │          │ │
│  └────────────┴───────────┴──────────┘ │
│                                       │
│  Uptime: 99.8%                        │
│  Last Error: None (24h)               │
│                                       │
└───────────────────────────────────────┘""",
            classes="api-section"
        )
    
    def create_keys_section(self) -> Static:
        """Active API keys section"""
        return Static(
            """┌─ 🔑 ACTIVE API KEYS ──────────────┐
│                                       │
│  ┌─ Name ─────┬─ Created ─┬─ Usage ─┐ │
│  │ 📱 Mobile │ 2 days ago│ 87 req  │ │
│  │ 🖥️  Dashboard1 week ago│ 42 req  │ │
│  │ 🔧 CLI    │ 1 mo ago  │ 13 req  │ │
│  │           │           │         │ │
│  └───────────┴───────────┴─────────┘ │
│                                       │
│  Total Keys: 3                       │
│  Last Used: 5 minutes ago            │
│  Expired: 0                          │
│                                       │
│  [C]reate  [R]evoke  [V]iew All      │
│                                       │
└───────────────────────────────────────┘""",
            classes="api-section"
        )
    
    def create_security_section(self) -> Static:
        """Security settings section"""
        return Static(
            """┌─ 🛡️ SECURITY SETTINGS ────────────┐
│                                       │
│  ● Authentication:  🔒 REQUIRED      │
│  ● Rate Limiting:   ✅ ENABLED       │
│  ● IP Whitelist:    ❌ DISABLED      │
│  ● HTTPS:           ❌ DISABLED      │
│  ● CORS:            ✅ ENABLED       │
│  ● Audit Logging:   ✅ ENABLED       │
│                                       │
│  Security Score:    85/100           │
│  Last Scan:         1 hour ago       │
│  Issues Found:      0                │
│                                       │
│  [E]dit Settings  [S]can Now         │
│                                       │
└───────────────────────────────────────┘""",
            classes="api-section"
        )
    
    def create_usage_section(self) -> Static:
        """Usage statistics section"""
        return Static(
            """┌─ 📊 USAGE STATISTICS ─────────────┐
│                                       │
│  ┌─────────────────────────────────┐ │
│  │ ███████████████████████████████ │ │
│  │ Requests per Hour (Last 24h)    │ │
│  └─────────────────────────────────┘ │
│                                       │
│  Peak: 12 req/min (14:30)           │
│  Average: 5.9 req/min               │
│  Total: 8,520 requests              │
│                                       │
│  Top Endpoints:                     │
│  1. /wallet/balance (42%)           │
│  2. /blockchain/status (23%)        │
│  3. /staking/info (15%)             │
│                                       │
└───────────────────────────────────────┘""",
            classes="api-section"
        )
    
    def create_limits_section(self) -> Static:
        """Rate limiting section"""
        return Static(
            """┌─ ⚡ RATE LIMITING ────────────────┐
│                                       │
│  Global Limits:                      │
│  ● Requests/Hour:  1,000             │
│  ● Requests/Min:   100               │
│  ● Burst:          50                │
│                                       │
│  Per-Key Limits:                     │
│  ● Requests/Hour:  500               │
│  ● Requests/Min:   50                │
│  ● Burst:          25                │
│                                       │
│  Current Usage:                      │
│  ● This Hour:      142/1,000         │
│  ● This Minute:    12/100            │
│  ● Status:         🟢 NORMAL         │
│                                       │
│  [A]djust Limits  [V]iew Logs        │
│                                       │
└───────────────────────────────────────┘""",
            classes="api-section"
        )
    
    def create_actions_section(self) -> Static:
        """Quick actions section"""
        return Static(
            """┌─ 🚀 QUICK ACTIONS ────────────────┐
│                                       │
│  ┌─────────────┬───────────────────┐ │
│  │ 🆕 Create   │  Generate new API │ │
│  │             │  key              │ │
│  ├─────────────┼───────────────────┤ │
│  │ 🗑️  Revoke   │  Remove API key  │ │
│  │             │  (immediate)      │ │
│  ├─────────────┼───────────────────┤ │
│  │ 📋 Logs     │  View API access  │ │
│  │             │  logs             │ │
│  ├─────────────┼───────────────────┤ │
│  │ ⚙️  Settings │  Configure API    │ │
│  │             │  security         │ │
│  └─────────────┴───────────────────┘ │
│                                       │
│  [C]reate [R]evoke [L]ogs [S]ettings │
│                                       │
└───────────────────────────────────────┘""",
            classes="api-section"
        )
    
    def on_key(self, event: events.Key) -> None:
        """Handle API management shortcuts"""
        key = event.key.lower()
        
        if key == "escape" or key == "b":
            self.app.pop_screen()
        elif key == "c":
            self.create_api_key()
        elif key == "r":
            self.revoke_api_key()
        elif key == "l":
            self.view_api_logs()
        elif key == "s":
            self.configure_settings()
        elif key == "1":
            self.create_api_key()
        elif key == "2":
            self.revoke_api_key()
    
    def create_api_key(self):
        """Create new API key"""
        try:
            # Implementation would use client.generate_api_key()
            self.notify("🆕 Create new API key interface")
        except Exception as e:
            self.notify(f"❌ API key creation failed: {e}")
    
    def revoke_api_key(self):
        """Revoke API key"""
        try:
            self.notify("🗑️ Revoke API key interface")
        except Exception as e:
            self.notify(f"❌ API key revocation failed: {e}")
    
    def view_api_logs(self):
        """View API access logs"""
        try:
            self.notify("📋 API access logs interface")
        except Exception as e:
            self.notify(f"❌ Failed to load logs: {e}")
    
    def configure_settings(self):
        """Configure API settings"""
        try:
            self.notify("⚙️ API settings configuration interface")
        except Exception as e:
            self.notify(f"❌ Settings configuration failed: {e}")