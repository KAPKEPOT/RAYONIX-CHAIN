#!/usr/bin/env python3
"""
Staking Management Screen
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Static, Button, DataTable, ProgressBar, Sparkline
from textual.screen import Screen
from textual import events
from typing import List, Dict, Any

class StakingScreen(Screen):
    """Comprehensive Staking Management"""
    
    CSS = """
    StakingScreen {
        align: center middle;
        background: #0f0f23;
    }
    
    #staking-container {
        grid-size: 2 3;
        grid-gutter: 1 2;
        padding: 1 2;
        height: 100%;
    }
    
    .staking-section {
        background: #1a1b26;
        border: round #2a2b3c;
        padding: 1 2;
    }
    
    .validator-card {
        background: #16161e;
        border: solid #2a2b3c;
        padding: 1;
        margin: 1 0;
    }
    
    .validator-card:hover {
        background: #2a2b3c;
        border: solid #7aa2f7;
    }
    
    .my-validator {
        border: double #9ece6a;
        background: #1a1b26;
    }
    
    .reward-positive {
        color: #9ece6a;
    }
    
    .reward-negative {
        color: #f7768e;
    }
    
    .performance-excellent {
        color: #9ece6a;
    }
    
    .performance-good {
        color: #e0af68;
    }
    
    .performance-poor {
        color: #f7768e;
    }
    """
    
    def __init__(self, rpc_client, app):
        super().__init__()
        self.client = rpc_client
        self.app = app
        self.selected_validator = None
    
    def compose(self) -> ComposeResult:
        yield Container(
            Grid(
                # My Staking Overview
                self.create_my_staking_section(),
                
                # Validator Performance
                self.create_performance_section(),
                
                # Quick Actions
                self.create_actions_section(),
                
                # Validator Rankings
                self.create_validators_section(),
                
                # Rewards & Earnings
                self.create_rewards_section(),
                
                # Delegation Management
                self.create_delegation_section(),
                
                id="staking-container"
            )
        )
    
    def create_my_staking_section(self) -> Static:
        """My staking overview section"""
        return Static(
            """┌─ ⚡ MY STAKING ────────────────────┐
│                                       │
│  Status:      🟢 ACTIVE              │
│  Rank:        #47 of 2,148           │
│  Uptime:      99.8%                  │
│                                       │
│  ┌─────────┬─────────┬─────────────┐ │
│  │ Staked  │ Rewards │ APR         │ │
│  │ 250.50  │  1.75   │   8.5%      │ │
│  │  RYX    │  RYX    │             │ │
│  └─────────┴─────────┴─────────────┘ │
│                                       │
│  Next Reward: 1h 23m                 │
│  Estimated:   0.058 RYX              │
│                                       │
└───────────────────────────────────────┘""",
            classes="staking-section"
        )
    
    def create_performance_section(self) -> Static:
        """Validator performance section"""
        return Static(
            """┌─ 📊 PERFORMANCE ──────────────────┐
│                                       │
│  ┌─────────────────────────────────┐ │
│  │ ███████████████████████████████ │ │
│  │ Last 30 Days Performance        │ │
│  └─────────────────────────────────┘ │
│                                       │
│  ● Blocks Proposed:   142            │
│  ● Blocks Signed:     8,542          │
│  ● Success Rate:      100%           │
│  ● Average Latency:   128ms          │
│                                       │
│  Performance: 🟢 EXCELLENT           │
│                                       │
└───────────────────────────────────────┘""",
            classes="staking-section"
        )
    
    def create_actions_section(self) -> Static:
        """Quick actions section"""
        return Static(
            """┌─ 🚀 QUICK ACTIONS ────────────────┐
│                                       │
│  ┌─────────────┬───────────────────┐ │
│  │ 🎯 Delegate │  Stake more funds │ │
│  │             │  to validator     │ │
│  ├─────────────┼───────────────────┤ │
│  │ 📤 Undelegate│  Unstake funds   │ │
│  │             │  from validator   │ │
│  ├─────────────┼───────────────────┤ │
│  │ 💰 Claim    │  Claim staking    │ │
│  │             │  rewards          │ │
│  ├─────────────┼───────────────────┤ │
│  │ 👑 Register │  Become a         │ │
│  │             │  validator        │ │
│  └─────────────┴───────────────────┘ │
│                                       │
│  [D]elegate [U]ndelegate [C]laim     │
│                                       │
└───────────────────────────────────────┘""",
            classes="staking-section"
        )
    
    def create_validators_section(self) -> Static:
        """Validator rankings section"""
        return Static(
            """┌─ 🏆 VALIDATOR RANKINGS ───────────┐
│                                       │
│  ┌─ Rank ─┬─ Validator ─┬─ Stake ───┐ │
│  │   1    │ ryx1valA... │ 50,000   │ │
│  │   2    │ ryx1valB... │ 45,000   │ │
│  │   3    │ ryx1valC... │ 40,000   │ │
│  │  ...   │     ...     │   ...    │ │
│  │   47   │ [YOU]       │   250    │ │
│  │   48   │ ryx1valD... │   240    │ │
│  └────────┴─────────────┴──────────┘ │
│                                       │
│  Total Validators: 2,148             │
│  Active: 1,892                       │
│  Average Fee: 5.2%                   │
│                                       │
└───────────────────────────────────────┘""",
            classes="staking-section"
        )
    
    def create_rewards_section(self) -> Static:
        """Rewards and earnings section"""
        return Static(
            """┌─ 💎 REWARDS & EARNINGS ───────────┐
│                                       │
│  Total Earned:     1.75 RYX          │
│  Available:        1.75 RYX          │
│  Pending:          0.25 RYX          │
│  Estimated APR:    8.5%              │
│                                       │
│  ┌─ Period ─┬─ Amount ─┬─ Growth ──┐ │
│  │ Today    │ 0.058    │ ↗ 5.8%    │ │
│  │ Week     │ 0.406    │ ↗ 4.1%    │ │
│  │ Month    │ 1.75     │ ↗ 3.9%    │ │
│  │ Year     │ 21.29    │ ↗ 8.5%    │ │
│  └──────────┴──────────┴────────────┘ │
│                                       │
│  [C]laim Rewards  [H]istory          │
│                                       │
└───────────────────────────────────────┘""",
            classes="staking-section"
        )
    
    def create_delegation_section(self) -> Static:
        """Delegation management section"""
        return Static(
            """┌─ 🔄 DELEGATION MANAGEMENT ────────┐
│                                       │
│  Current Delegation:                 │
│  ● Validator:    [SELF]              │
│  ● Amount:       250.50 RYX          │
│  ● Duration:     45 days             │
│  ● Unlock Time:  15 days from now    │
│                                       │
│  Available for Delegation:           │
│  ● Balance:      1,000.25 RYX        │
│  ● Min Stake:    1.0 RYX             │
│  ● Max Stake:    No limit            │
│                                       │
│  [D]elegate More  [U]ndelegate       │
│  [S]witch Validator                  │
│                                       │
└───────────────────────────────────────┘""",
            classes="staking-section"
        )
    
    def on_key(self, event: events.Key) -> None:
        """Handle staking keyboard shortcuts"""
        key = event.key.lower()
        
        if key == "escape" or key == "b":
            self.app.pop_screen()
        elif key == "d":
            self.delegate_funds()
        elif key == "u":
            self.undelegate_funds()
        elif key == "c":
            self.claim_rewards()
        elif key == "r":
            self.register_validator()
        elif key == "1":
            self.delegate_funds()
        elif key == "2":
            self.undelegate_funds()
        elif key == "3":
            self.claim_rewards()
    
    def delegate_funds(self):
        """Delegate funds to validator"""
        try:
            # Implementation would use client.stake_tokens()
            self.notify("🎯 Delegate funds interface")
        except Exception as e:
            self.notify(f"❌ Delegation failed: {e}")
    
    def undelegate_funds(self):
        """Undelegate funds from validator"""
        try:
            self.notify("📤 Undelegate funds interface")
        except Exception as e:
            self.notify(f"❌ Undelegation failed: {e}")
    
    def claim_rewards(self):
        """Claim staking rewards"""
        try:
            self.notify("💰 Claim rewards interface")
        except Exception as e:
            self.notify(f"❌ Reward claim failed: {e}")
    
    def register_validator(self):
        """Register as validator"""
        try:
            self.notify("👑 Validator registration interface")
        except Exception as e:
            self.notify(f"❌ Validator registration failed: {e}")