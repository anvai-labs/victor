#!/usr/bin/env bash
# Victor Installation Script for macOS and Linux
# Copyright 2025 Vijaykumar Singh
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/vijayksingh/victor/main/scripts/install/install.sh | bash
#
# Or with options:
#   curl -fsSL ... | bash -s -- --dev     # Include dev dependencies
#   curl -fsSL ... | bash -s -- --binary  # Install standalone binary
#   curl -fsSL ... | bash -s -- --pipx    # Use pipx (isolated)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
INSTALL_TYPE="pip"
INSTALL_DEV=false
VICTOR_VERSION="latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --binary)
            INSTALL_TYPE="binary"
            shift
            ;;
        --pipx)
            INSTALL_TYPE="pipx"
            shift
            ;;
        --version)
            VICTOR_VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Victor Installation Script"
            echo ""
            echo "Usage: install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev       Include development dependencies"
            echo "  --binary    Install standalone binary (no Python required)"
            echo "  --pipx      Use pipx for isolated installation"
            echo "  --version   Specify version (default: latest)"
            echo "  --help      Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║   ██╗   ██╗██╗ ██████╗████████╗ ██████╗ ██████╗          ║"
echo "║   ██║   ██║██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗         ║"
echo "║   ██║   ██║██║██║        ██║   ██║   ██║██████╔╝         ║"
echo "║   ╚██╗ ██╔╝██║██║        ██║   ██║   ██║██╔══██╗         ║"
echo "║    ╚████╔╝ ██║╚██████╗   ██║   ╚██████╔╝██║  ██║         ║"
echo "║     ╚═══╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝         ║"
echo "║                                                           ║"
echo "║          Enterprise-Ready AI Coding Assistant             ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Detect OS
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo -e "${GREEN}➤ Detected: ${OS} (${ARCH})${NC}"
echo ""

# Check for Python
check_python() {
    local candidate
    for candidate in python3 python3.13 python3.12; do
        if command -v "$candidate" &> /dev/null &&
            "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' 2>/dev/null; then
            VICTOR_PYTHON=$(command -v "$candidate")
            echo -e "${GREEN}✓ $("$VICTOR_PYTHON" --version) found${NC}"
            return 0
        fi
    done
    echo -e "${RED}✗ Python 3.12+ required; install it and activate its environment.${NC}"
    return 1
}

# Install Python if needed
install_python() {
    echo -e "${YELLOW}Installing Python...${NC}"

    if [[ "$OS" == "darwin" ]]; then
        if command -v brew &> /dev/null; then
            brew install python@3.12
        else
            echo -e "${RED}Homebrew not found. Please install Python 3.12+ manually.${NC}"
            echo "Visit: https://www.python.org/downloads/"
            exit 1
        fi
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y python3.12 python3.12-venv python3-pip
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3.12
        elif command -v pacman &> /dev/null; then
            sudo pacman -S python
        else
            echo -e "${RED}Unable to install Python. Please install Python 3.12+ manually.${NC}"
            exit 1
        fi
    fi
}

# Install via pip
install_pip() {
    echo -e "${BLUE}➤ Installing Victor via pip...${NC}"

    if [[ "$INSTALL_DEV" == true ]]; then
        "$VICTOR_PYTHON" -m pip install --user "victor-ai[dev]"
    else
        "$VICTOR_PYTHON" -m pip install --user victor-ai
    fi

    echo -e "${GREEN}✓ Victor installed successfully!${NC}"
}

# Install via pipx
install_pipx() {
    echo -e "${BLUE}➤ Installing Victor via pipx (isolated)...${NC}"

    # Check if pipx is installed
    if ! command -v pipx &> /dev/null; then
        echo -e "${YELLOW}Installing pipx first...${NC}"
        "$VICTOR_PYTHON" -m pip install --user pipx
        "$VICTOR_PYTHON" -m pipx ensurepath
    fi

    pipx install --python "$VICTOR_PYTHON" victor-ai

    echo -e "${GREEN}✓ Victor installed successfully!${NC}"
}

# Install standalone binary
install_binary() {
    echo -e "${BLUE}➤ Installing Victor binary...${NC}"

    # Determine platform
    if [[ "$OS" == "darwin" ]]; then
        PLATFORM="macos"
    else
        PLATFORM="linux"
    fi

    if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
        ARCH_SUFFIX="arm64"
    else
        ARCH_SUFFIX="x64"
    fi

    BINARY_NAME="victor-${PLATFORM}-${ARCH_SUFFIX}"
    DOWNLOAD_URL="https://github.com/vijayksingh/victor/releases/latest/download/${BINARY_NAME}.tar.gz"

    echo -e "${BLUE}Downloading ${BINARY_NAME}...${NC}"

    INSTALL_DIR="${HOME}/.local/bin"
    mkdir -p "$INSTALL_DIR"

    # Download and extract
    curl -fsSL "$DOWNLOAD_URL" -o /tmp/victor.tar.gz
    tar -xzf /tmp/victor.tar.gz -C "$INSTALL_DIR"
    chmod +x "${INSTALL_DIR}/victor"
    rm /tmp/victor.tar.gz

    # Add to PATH if needed
    if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
        echo -e "${YELLOW}Adding ${INSTALL_DIR} to PATH...${NC}"

        if [[ -f "${HOME}/.zshrc" ]]; then
            echo 'export PATH="${HOME}/.local/bin:$PATH"' >> "${HOME}/.zshrc"
        elif [[ -f "${HOME}/.bashrc" ]]; then
            echo 'export PATH="${HOME}/.local/bin:$PATH"' >> "${HOME}/.bashrc"
        fi

        export PATH="${INSTALL_DIR}:$PATH"
    fi

    echo -e "${GREEN}✓ Victor binary installed to ${INSTALL_DIR}/victor${NC}"
}

# Main installation
main() {
    case $INSTALL_TYPE in
        pip)
            if ! check_python; then
                install_python
                check_python || exit 1
            fi
            install_pip
            ;;
        pipx)
            if ! check_python; then
                install_python
                check_python || exit 1
            fi
            install_pipx
            ;;
        binary)
            install_binary
            ;;
    esac

    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              Installation Complete!                       ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Quick Start:"
    echo ""
    echo "  1. Initialize Victor in your project:"
    echo -e "     ${BLUE}victor init${NC}"
    echo ""
    echo "  2. Start chatting:"
    echo -e "     ${BLUE}victor chat${NC}"
    echo ""
    echo "  3. Or use the TUI:"
    echo -e "     ${BLUE}victor${NC}"
    echo ""
    echo "For more information, visit:"
    echo "  https://github.com/vijayksingh/victor"
    echo ""
}

main
