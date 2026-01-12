# PowerShell version of benchmark_proofs.sh
# benchmark_proofs.ps1 - Benchmark Noir circuit proof generation
# Usage: .\benchmark_proofs.ps1

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "NOIR CIRCUIT PROOF GENERATION BENCHMARK" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Circuit names and expected times
$circuits = @(
    @{Name="aggregate_votes"; Expected=0.05},
    @{Name="byzantine_detector"; Expected=0.08},
    @{Name="reputation_updater"; Expected=0.03},
    @{Name="shamir_verifier"; Expected=0.04},
    @{Name="round_storage"; Expected=0.02}
)

Write-Host "Running proof generation benchmarks..." -ForegroundColor Yellow
Write-Host ""

$totalTime = 0
$totalSize = 0

foreach ($circuit in $circuits) {
    $name = $circuit.Name
    $expected = $circuit.Expected
    
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "Testing: $name" -ForegroundColor White
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    
    # Run 5 iterations
    $sum = 0
    for ($i = 1; $i -le 5; $i++) {
        $start = Get-Date
        nargo prove $name *>$null
        $end = Get-Date
        $elapsed = ($end - $start).TotalSeconds
        $sum += $elapsed
        Write-Host "  Iteration $i: $($elapsed.ToString('F4'))s"
    }
    
    # Calculate average
    $avg = $sum / 5
    $totalTime += $avg
    
    # Check if within expected range (±20%)
    $lower = $expected * 0.8
    $upper = $expected * 1.2
    
    if ($avg -ge $lower -and $avg -le $upper) {
        Write-Host "  ✓ Average: $($avg.ToString('F4'))s (expected: ${expected}s)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Average: $($avg.ToString('F4'))s (expected: ${expected}s - OUTSIDE RANGE)" -ForegroundColor Yellow
    }
    
    # Get proof size
    $proofPath = "$env:TEMP\${name}.proof"
    if (Test-Path $proofPath) {
        $size = (Get-Item $proofPath).Length
        $sizeKb = [math]::Round($size / 1024, 2)
        $totalSize += $size
        Write-Host "  Proof size: ${sizeKb}KB"
    }
    
    Write-Host ""
}

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "Total proof generation time: $($totalTime.ToString('F4'))s"
$totalSizeKb = [math]::Round($totalSize / 1024, 2)
Write-Host "Total proof size: ${totalSizeKb}KB"
Write-Host ""

# Check if total time meets target (<250ms = 0.25s)
if ($totalTime -lt 0.25) {
    Write-Host "✓ Performance target met! (<250ms)" -ForegroundColor Green
} else {
    Write-Host "⚠ Performance target not met ($($totalTime.ToString('F4'))s > 250ms)" -ForegroundColor Yellow
}

# Check if total size meets target (<6KB)
if ($totalSizeKb -lt 6) {
    Write-Host "✓ Size target met! (<6KB)" -ForegroundColor Green
} else {
    Write-Host "⚠ Size target not met (${totalSizeKb}KB > 6KB)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Benchmark complete!" -ForegroundColor Cyan
