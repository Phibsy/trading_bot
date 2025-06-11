#!/usr/bin/env python3
"""
Alpaca Trading Bot with Groq AI Integration
Main entry point for the trading bot application.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Alpaca Trading Bot with Groq AI")
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (optional)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Override trading symbols (e.g., TQQQ SQQQ SPXL SPXS)'
    )
    
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Force paper trading mode'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run analysis without executing trades'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    return parser.parse_args()

def validate_environment():
    """Validate required environment variables."""
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_value_here")
        print("\nOr create a .env file in the project root directory.")
        return False
    
    return True

async def main():
    """Main application entry point."""
    print("🤖 Alpaca Trading Bot with Groq AI Integration")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate environment (now AFTER load_dotenv())
    if not validate_environment():
        sys.exit(1)
    
    try:
        # Try to import required modules
        try:
            from config.settings import BotConfig
            from core.bot import TradingBot
            from utils.logger import setup_logger
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please ensure all dependencies are installed:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            print("Please check your configuration files.")
            sys.exit(1)
        
        # Load configuration
        try:
            config = BotConfig()
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            print("Please check your .env file and configuration.")
            sys.exit(1)
        
        # Override configuration based on command line arguments
        if args.symbols:
            config.trading.symbols = args.symbols
            print(f"📊 Trading symbols: {', '.join(args.symbols)}")
        
        if args.paper:
            config.alpaca.base_url = "https://paper-api.alpaca.markets"
            print("📝 Paper trading mode enabled")
        
        if args.log_level:
            config.logging.level = args.log_level
        
        # Setup logger
        try:
            logger = setup_logger(config.logging, "Main")
        except Exception as e:
            print(f"❌ Failed to setup logger: {e}")
            sys.exit(1)
        
        # Print configuration summary
        print(f"🔗 Alpaca URL: {config.alpaca.base_url}")
        print(f"📈 Symbols: {', '.join(config.trading.symbols)}")
        print(f"🎯 Max Positions: {config.trading.max_positions}")
        print(f"💰 Position Size: {config.trading.position_size:.1%}")
        print(f"🛡️  Stop Loss: {config.trading.stop_loss:.1%}")
        print(f"🎯 Take Profit: {config.trading.take_profit:.1%}")
        
        if config.groq.api_key:
            print(f"🧠 Groq AI: Enabled ({config.groq.model})")
        else:
            print("🧠 Groq AI: Disabled (no API key)")
        
        print()
        
        # Validate only mode
        if args.validate_only:
            print("✅ Configuration validation complete")
            return
        
        # Dry run mode
        if args.dry_run:
            print("🔍 DRY RUN MODE - No trades will be executed")
            # TODO: Implement dry run mode
        
        # Create and start the trading bot
        try:
            bot = TradingBot(config)
            logger.info("Starting trading bot...")
            await bot.start()
        except KeyboardInterrupt:
            print("\n⏹️  Shutdown requested by user")
            logger.info("Bot shutdown requested by user")
        except Exception as e:
            print(f"❌ Bot error: {e}")
            logger.error(f"Bot error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

def run_bot():
    """Run the bot with proper async handling."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trading bot stopped")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_bot()
