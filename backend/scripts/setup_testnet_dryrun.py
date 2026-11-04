#!/usr/bin/env python3
"""
Testnet Dry Run Setup Script

Creates minimal DB entities for a testnet execution cycle:
1. User + Account (AI Trader)
2. Hyperliquid testnet wallet (encrypted key from .env)
3. SystemConfig: global_trading_mode = testnet
4. TradingProgram with compiled strategy code
5. AccountProgramBinding with scheduled trigger (interval=60s)

Run:  cd ~/Projects/Hyper-Alpha-Arena/backend && .venv/bin/python scripts/setup_testnet_dryrun.py
"""

import os
import sys
import json
from pathlib import Path

# Load .env first
from dotenv import load_dotenv
load_dotenv()

# Ensure DATABASE_URL points to local DB
os.environ.setdefault("DATABASE_URL", "postgresql://alpha_user:alpha_pass@127.0.0.1:5432/alpha_arena")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import SessionLocal
from database.models import (
    User, Account, SystemConfig, HyperliquidWallet,
    TradingProgram, AccountProgramBinding
)
from backtest.ab_validators import exploration_validator
from utils.encryption import encrypt_private_key


def main():
    db = SessionLocal()
    try:
        # ------------------------------------------------------------------
        # 1. User (only needed for FK)
        # ------------------------------------------------------------------
        user = db.query(User).filter_by(username="testnet_dryrun").first()
        if not user:
            user = User(
                username="testnet_dryrun",
                email="testnet@dryrun.local",
                password_hash="unused",  # no auth needed
                is_active="true",
            )
            db.add(user)
            db.flush()
            print(f"Created user: id={user.id}")
        else:
            print(f"User exists: id={user.id}")

        # ------------------------------------------------------------------
        # 2. Account (AI Trader)
        # ------------------------------------------------------------------
        account = db.query(Account).filter_by(name="Testnet Dryrun").first()
        if not account:
            account = Account(
                user_id=user.id,
                name="Testnet Dryrun",
                account_type="AI",
                is_active="true",
                auto_trading_enabled="true",
                hyperliquid_environment="testnet",
                max_leverage=3,
                default_leverage=1,
                initial_capital=10000.0,
            )
            db.add(account)
            db.flush()
            print(f"Created account: id={account.id}")
        else:
            print(f"Account exists: id={account.id}")

        # ------------------------------------------------------------------
        # 3. SystemConfig: global_trading_mode = testnet
        # ------------------------------------------------------------------
        cfg = db.query(SystemConfig).filter_by(key="global_trading_mode").first()
        if not cfg:
            cfg = SystemConfig(
                key="global_trading_mode",
                value="testnet",
                description="Global trading mode for all AI Traders (paper/testnet/mainnet)",
            )
            db.add(cfg)
            print("Created system_config: global_trading_mode=testnet")
        else:
            cfg.value = "testnet"
            print("Updated system_config: global_trading_mode=testnet")

        # ------------------------------------------------------------------
        # 4. Hyperliquid testnet wallet (encrypted key from .env)
        # ------------------------------------------------------------------
        testnet_key = os.environ.get("HYPERLIQUID_TESTNET_PRIVATE_KEY")
        testnet_wallet_addr = os.environ.get("HYPERLIQUID_TESTNET_WALLET")
        enc_key_env = os.environ.get("HYPERLIQUID_ENCRYPTION_KEY")

        if not testnet_key:
            raise RuntimeError("HYPERLIQUID_TESTNET_PRIVATE_KEY not in .env")
        if not testnet_wallet_addr:
            raise RuntimeError("HYPERLIQUID_TESTNET_WALLET not in .env")
        if not enc_key_env:
            raise RuntimeError("HYPERLIQUID_ENCRYPTION_KEY not in .env")

        wallet = db.query(HyperliquidWallet).filter_by(
            account_id=account.id, environment="testnet"
        ).first()

        encrypted_key = encrypt_private_key(testnet_key)

        if not wallet:
            wallet = HyperliquidWallet(
                account_id=account.id,
                environment="testnet",
                private_key_encrypted=encrypted_key,
                wallet_address=testnet_wallet_addr,
                max_leverage=3,
                default_leverage=1,
                is_active="true",
                key_type="private_key",
            )
            db.add(wallet)
            print(f"Created testnet wallet: {testnet_wallet_addr}")
        else:
            wallet.private_key_encrypted = encrypted_key
            wallet.wallet_address = testnet_wallet_addr
            wallet.is_active = "true"
            wallet.max_leverage = 3
            wallet.default_leverage = 1
            print(f"Updated testnet wallet: {testnet_wallet_addr}")

        # ------------------------------------------------------------------
        # 5. TradingProgram with compiled strategy code
        # ------------------------------------------------------------------
        # We'll use a minimal EMA cross + Donchian + RSI band strategy
        # compiled from the rules we tested earlier
        strategy_code = '''"""
EMA cross + Donchian breakout + RSI band (compiled from LLM rules).
Generated by trade_agent_compiler.compile_prompt_rules.
"""
from program_trader.models import MarketData, Decision

class Strategy:
    def __init__(self):
        self.params = {}

    def init(self, params: dict) -> None:
        self.params = params

    def should_trade(self, data: MarketData) -> Decision:
        symbol = data.trigger_symbol or "BTC"
        price = data.get_price(symbol)
        if price <= 0:
            return Decision(operation="hold", symbol=symbol, reason="no price")

        # Indicators - use exact keys from calculate_indicators output
        ema_fast = data.get_indicator(symbol, "EMA12", "1h").get("value", 0)
        ema_slow = data.get_indicator(symbol, "EMA55", "1h").get("value", 0)
        donchian = data.get_indicator(symbol, "DONCHIAN55", "1h")
        donchian_high = donchian.get("upper", 0)
        rsi = data.get_indicator(symbol, "RSI14", "1h").get("value", 50)
        # Rules (from LLM rules: ema12>ema55, close>55-high*0.99, rsi 30-70)

        long_cond = (
            ema_fast > ema_slow and
            price > donchian_high * 0.99 and
            30 <= rsi <= 70
        )

        positions = data.positions or {}
        has_position = symbol in positions and abs(positions[symbol].size) > 0

        if long_cond and not has_position:
            return Decision(
                operation="buy",
                symbol=symbol,
                target_portion_of_balance=0.35,
                leverage=2,
                max_price=price * 1.001,
                reason="ema cross + donchian breakout + rsi band",
                trading_strategy="EMA12>55 + Donchian 55h*0.99 + RSI 30-70",
            )

        if has_position:
            pos = positions[symbol]
            if pos.side == "long":
                # Simple exit: EMA cross down
                if ema_fast < ema_slow:
                    return Decision(
                        operation="close",
                        symbol=symbol,
                        target_portion_of_balance=1.0,
                        leverage=pos.leverage,
                        min_price=price * 0.999,
                        reason="ema cross down",
                    )

        return Decision(operation="hold", symbol=symbol, reason="no signal")'''

        # ------------------------------------------------------------------
        # Performance Gate: validate strategy code before deployment
        # ------------------------------------------------------------------
        print("Running exploration_validator on strategy code...")
        passed, details = exploration_validator(
            db, strategy_code, max_triggers=8000, backtest_days=365,
            symbols=["BTC"]
        )
        if not passed:
            raise RuntimeError(
                f"Strategy FAILED performance gate (exploration threshold). "
                f"Details: {details}"
            )
        print("Performance gate PASSED")

        program = db.query(TradingProgram).filter_by(name="EMA-Donchian-RSI Testnet").first()
        if not program:
            program = TradingProgram(
                user_id=user.id,
                name="EMA-Donchian-RSI Testnet",
                description="Compiled from LLM rules: EMA12/55 cross + Donchian 55h breakout *0.99 + RSI 30-70 band",
                code=strategy_code,
                params=json.dumps({
                    "ema_fast": 12,
                    "ema_slow": 55,
                    "donchian_lookback": 55,
                    "donchian_threshold": 0.99,
                    "rsi_low": 30,
                    "rsi_high": 70,
                    "portion": 0.35,
                    "leverage": 2,
                }),
                is_deleted=False,
            )
            db.add(program)
            db.flush()
            print(f"Created program: id={program.id}")
        else:
            program.code = strategy_code
            program.params = json.dumps({
                "ema_fast": 12,
                "ema_slow": 55,
                "donchian_lookback": 55,
                "donchian_threshold": 0.99,
                "rsi_low": 30,
                "rsi_high": 70,
                "portion": 0.35,
                "leverage": 2,
            })
            print(f"Updated program: id={program.id}")

        # ------------------------------------------------------------------
        # 6. AccountProgramBinding (scheduled trigger, 60s interval)
        # ------------------------------------------------------------------
        binding = db.query(AccountProgramBinding).filter_by(
            account_id=account.id, program_id=program.id
        ).first()

        if not binding:
            binding = AccountProgramBinding(
                account_id=account.id,
                program_id=program.id,
                signal_pool_ids="[]",
                trigger_interval=60,  # 60 seconds
                scheduled_trigger_enabled=True,
                is_active=True,
                exchange="hyperliquid",
                params_override=json.dumps({}),
            )
            db.add(binding)
            db.flush()
            print(f"Created binding: id={binding.id} (interval=60s, scheduled=True)")
        else:
            binding.trigger_interval = 60
            binding.scheduled_trigger_enabled = True
            binding.is_active = True
            binding.exchange = "hyperliquid"
            print(f"Updated binding: id={binding.id}")

        db.commit()
        print("\n✅ Testnet dry run setup complete!")
        print(f"  Account: {account.name} (id={account.id})")
        print(f"  Program: {program.name} (id={program.id})")
        print(f"  Binding: id={binding.id} — will trigger every 60s")
        print(f"  Wallet: {testnet_wallet_addr} (testnet)")
        print(f"  Mode: global_trading_mode=testnet")
        print("\nNext: start the program execution service and watch logs.")
        print("  cd ~/Projects/Hyper-Alpha-Arena/backend")
        print("  .venv/bin/python -m services.program_execution_service")

    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()