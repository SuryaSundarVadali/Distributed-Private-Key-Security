#!/bin/bash
# setup.sh - Setup Aztec + Noir + Barretenberg environment
# Usage: ./setup.sh

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DeFi Agent Coordination - Aztec + Noir Setup               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

# Step 1: Check prerequisites
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Checking Prerequisites"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if command_exists curl; then
    print_status 0 "curl installed"
else
    print_status 1 "curl not found - please install curl"
    exit 1
fi

if command_exists git; then
    print_status 0 "git installed"
else
    print_status 1 "git not found - please install git"
    exit 1
fi

if command_exists node; then
    NODE_VERSION=$(node --version)
    print_status 0 "Node.js installed ($NODE_VERSION)"
else
    print_status 1 "Node.js not found - please install Node.js 18+"
    exit 1
fi

echo ""

# Step 2: Install Noir
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Installing Noir Compiler"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if command_exists nargo; then
    NARGO_VERSION=$(nargo --version | head -n1)
    print_status 0 "Noir already installed ($NARGO_VERSION)"
else
    echo "Installing noirup..."
    curl -L https://raw.githubusercontent.com/noir-lang/noirup/main/install.sh | bash
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    
    echo "Installing Noir..."
    noirup
    
    if command_exists nargo; then
        print_status 0 "Noir installed successfully"
    else
        print_status 1 "Noir installation failed"
        exit 1
    fi
fi

echo ""

# Step 3: Install Barretenberg backend
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Installing Barretenberg Backend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Setting Barretenberg as backend..."
nargo backend use barretenberg

if nargo backend list | grep -q "barretenberg"; then
    print_status 0 "Barretenberg backend configured"
else
    print_status 1 "Barretenberg backend configuration failed"
    exit 1
fi

echo ""

# Step 4: Install Aztec CLI
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Installing Aztec CLI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if command_exists aztec; then
    AZTEC_VERSION=$(aztec --version 2>/dev/null || echo "unknown")
    print_status 0 "Aztec CLI already installed ($AZTEC_VERSION)"
else
    echo "Installing Aztec CLI..."
    npm install -g @aztec/cli
    
    if command_exists aztec; then
        print_status 0 "Aztec CLI installed successfully"
    else
        print_status 1 "Aztec CLI installation failed"
        exit 1
    fi
fi

echo ""

# Step 5: Compile circuits
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Compiling Noir Circuits"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd circuits

echo "Compiling all circuits..."
nargo compile

if [ $? -eq 0 ]; then
    print_status 0 "All circuits compiled successfully"
else
    print_status 1 "Circuit compilation failed"
    exit 1
fi

cd ..

echo ""

# Step 6: Create example Prover.toml files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6: Creating Example Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create example Prover.toml for aggregate_votes
cat > circuits/Prover.toml << 'EOF'
# Example inputs for aggregate_votes circuit
# Modify these values for your use case

votes = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
threshold = 7
commitments = [
    "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    "0x2345678901bcdef02345678901bcdef02345678901bcdef02345678901bcdef0",
    "0x3456789012cdef013456789012cdef013456789012cdef013456789012cdef01",
    "0x456789023def0124567890123def0124567890123def0124567890123def012",
    "0x56789034ef012345678901234ef012345678901234ef012345678901234ef01",
    "0x6789045f0123456789012345f0123456789012345f0123456789012345f012",
    "0x789056012345678901234567012345678901234567012345678901234567012",
    "0x89067123456789012345678123456789012345678123456789012345678123",
    "0x90178234567890123456789234567890123456789234567890123456789234",
    "0x01289345678901234567890345678901234567890345678901234567890345"
]
expected_root = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd"
EOF

print_status 0 "Created example Prover.toml"

echo ""

# Step 7: Verify installation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 7: Verifying Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Running nargo check..."
cd circuits
nargo check

if [ $? -eq 0 ]; then
    print_status 0 "Noir circuits check passed"
else
    print_status 1 "Noir circuits check failed"
    cd ..
    exit 1
fi

cd ..

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   SETUP COMPLETE!                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Review circuits in ./circuits/"
echo "  2. Run tests: nargo test"
echo "  3. Generate proofs: nargo prove <circuit_name>"
echo "  4. Benchmark performance: ./scripts/benchmark_proofs.sh"
echo ""
echo "Documentation:"
echo "  - Full guide: AZTEC_NOIR_DEVELOPER_INTEGRATION.md"
echo "  - Quick start: README.md"
echo ""
