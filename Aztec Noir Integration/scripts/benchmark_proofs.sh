#!/bin/bash
# benchmark_proofs.sh - Benchmark Noir circuit proof generation
# Usage: ./benchmark_proofs.sh

set -e

echo "========================================="
echo "NOIR CIRCUIT PROOF GENERATION BENCHMARK"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Circuit names
CIRCUITS=("aggregate_votes" "byzantine_detector" "reputation_updater" "shamir_verifier" "round_storage")

# Expected times (in seconds)
EXPECTED_TIMES=(0.05 0.08 0.03 0.04 0.02)

echo "Running proof generation benchmarks..."
echo ""

total_time=0
total_size=0

for i in "${!CIRCUITS[@]}"; do
    circuit="${CIRCUITS[$i]}"
    expected="${EXPECTED_TIMES[$i]}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $circuit"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run 5 iterations
    sum=0
    for iter in {1..5}; do
        start=$(date +%s.%N)
        nargo prove "$circuit" > /dev/null 2>&1
        end=$(date +%s.%N)
        elapsed=$(echo "$end - $start" | bc)
        sum=$(echo "$sum + $elapsed" | bc)
        echo "  Iteration $iter: ${elapsed}s"
    done
    
    # Calculate average
    avg=$(echo "scale=4; $sum / 5" | bc)
    total_time=$(echo "$total_time + $avg" | bc)
    
    # Check if within expected range (±20%)
    lower=$(echo "$expected * 0.8" | bc)
    upper=$(echo "$expected * 1.2" | bc)
    
    if (( $(echo "$avg >= $lower" | bc -l) )) && (( $(echo "$avg <= $upper" | bc -l) )); then
        echo -e "  ${GREEN}✓ Average: ${avg}s (expected: ${expected}s)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Average: ${avg}s (expected: ${expected}s - OUTSIDE RANGE)${NC}"
    fi
    
    # Get proof size
    if [ -f "/tmp/${circuit}.proof" ]; then
        size=$(stat -f%z "/tmp/${circuit}.proof" 2>/dev/null || stat -c%s "/tmp/${circuit}.proof" 2>/dev/null)
        size_kb=$(echo "scale=2; $size / 1024" | bc)
        total_size=$(echo "$total_size + $size" | bc)
        echo "  Proof size: ${size_kb}KB"
    fi
    
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total proof generation time: ${total_time}s"
total_size_kb=$(echo "scale=2; $total_size / 1024" | bc)
echo "Total proof size: ${total_size_kb}KB"
echo ""

# Check if total time meets target (<250ms = 0.25s)
if (( $(echo "$total_time < 0.25" | bc -l) )); then
    echo -e "${GREEN}✓ Performance target met! (<250ms)${NC}"
else
    echo -e "${YELLOW}⚠ Performance target not met (${total_time}s > 250ms)${NC}"
fi

# Check if total size meets target (<6KB)
if (( $(echo "$total_size_kb < 6" | bc -l) )); then
    echo -e "${GREEN}✓ Size target met! (<6KB)${NC}"
else
    echo -e "${YELLOW}⚠ Size target not met (${total_size_kb}KB > 6KB)${NC}"
fi

echo ""
echo "Benchmark complete!"
