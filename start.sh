#!/bin/bash

# Alpaca Trading Bot Startup Script
# This script provides an easy way to start the trading bot with proper checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
check_python() {
    print_status "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(${PYTHON_CMD} --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION found"
}

# Check virtual environment
check_venv() {
    if [ -z "$VIRTUAL_ENV" ] && [ ! -d "venv" ]; then
        print_warning "No virtual environment detected"
        print_status "Creating virtual environment..."
        ${PYTHON_CMD} -m venv venv
        print_success "Virtual environment created"
    fi
    
    if [ -z "$VIRTUAL_ENV" ] && [ -d "venv" ]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate || source venv/Scripts/activate 2>/dev/null || {
            print_error "Failed to activate virtual environment"
            exit 1
        }
        print_success "Virtual environment activated"
    fi
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Check if dependencies are installed
    ${PYTHON_CMD} -c "
import sys
try:
    import alpaca_trade_api, pandas, numpy, ta, dotenv
    print('Core dependencies found')
except ImportError as e:
    print(f'Missing dependency: {e}')
    sys.exit(1)
" || {
        print_warning "Installing dependencies..."
        pip install -r requirements.txt
        print_success "Dependencies installed"
    }
}

# Check configuration
check_config() {
    print_status "Checking configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning "No .env file found. Creating from template..."
            cp .env.example .env
            print_warning "Please edit .env file with your API keys before running the bot"
            print_status "Opening .env file for editing..."
            ${EDITOR:-nano} .env
        else
            print_error "No .env or .env.example file found"
            exit 1
        fi
    fi
    
    # Check for required environment variables
    source .env 2>/dev/null || true
    
    if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
        print_error "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file"
        exit 1
    fi
    
    print_success "Configuration validated"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p logs data
    print_success "Directories created"
}

# Main startup function
start_bot() {
    print_status "Starting Alpaca Trading Bot..."
    echo "=================================="
    
    # Parse command line arguments
    MODE="normal"
    EXTRA_ARGS=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --paper)
                MODE="paper"
                EXTRA_ARGS="$EXTRA_ARGS --paper"
                shift
                ;;
            --debug)
                EXTRA_ARGS="$EXTRA_ARGS --log-level DEBUG"
                shift
                ;;
            --validate)
                MODE="validate"
                EXTRA_ARGS="$EXTRA_ARGS --validate-only"
                shift
                ;;
            --symbols)
                shift
                SYMBOLS=""
                while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                    SYMBOLS="$SYMBOLS $1"
                    shift
                done
                EXTRA_ARGS="$EXTRA_ARGS --symbols$SYMBOLS"
                ;;
            *)
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
                ;;
        esac
    done
    
    case $MODE in
        paper)
            print_warning "Starting in PAPER TRADING mode"
            ;;
        validate)
            print_status "Validation mode - no trading will occur"
            ;;
        normal)
            print_warning "Starting in LIVE TRADING mode"
            print_warning "This will execute real trades with real money!"
            read -p "Are you sure you want to continue? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_status "Aborted by user"
                exit 0
            fi
            ;;
    esac
    
    # Start the bot
    print_status "Executing: ${PYTHON_CMD} main.py${EXTRA_ARGS}"
    ${PYTHON_CMD} main.py${EXTRA_ARGS}
}

# Main execution
main() {
    echo "🤖 Alpaca Trading Bot Startup Script"
    echo "===================================="
    
    check_python
    check_venv
    check_dependencies
    check_config
    create_directories
    
    start_bot "$@"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --paper         Start in paper trading mode"
    echo "  --debug         Enable debug logging"
    echo "  --validate      Validate configuration only"
    echo "  --symbols LIST  Override trading symbols (e.g., --symbols TQQQ SQQQ)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --paper                    # Start in paper trading mode"
    echo "  $0 --debug --paper            # Debug mode with paper trading"
    echo "  $0 --symbols TQQQ SQQQ QQQ    # Trade specific symbols"
    echo "  $0 --validate                 # Just validate configuration"
}

# Check if help is requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
