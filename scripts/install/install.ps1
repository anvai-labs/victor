# Victor Installation Script for Windows
# Copyright 2025 Vijaykumar Singh
#
# Usage (PowerShell):
#   iwr -useb https://raw.githubusercontent.com/vijayksingh/victor/main/scripts/install/install.ps1 | iex
#
# Or with options:
#   .\install.ps1 -Dev          # Include dev dependencies
#   .\install.ps1 -Binary       # Install standalone binary
#   .\install.ps1 -Pipx         # Use pipx (isolated)

param(
    [switch]$Dev,
    [switch]$Binary,
    [switch]$Pipx,
    [string]$Version = "latest",
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Colors via Write-Host
function Write-Info { Write-Host $args[0] -ForegroundColor Blue }
function Write-Success { Write-Host $args[0] -ForegroundColor Green }
function Write-Warning { Write-Host $args[0] -ForegroundColor Yellow }
function Write-Error { Write-Host $args[0] -ForegroundColor Red }

if ($Help) {
    Write-Host @"
Victor Installation Script

Usage: install.ps1 [OPTIONS]

Options:
  -Dev       Include development dependencies
  -Binary    Install standalone binary (no Python required)
  -Pipx      Use pipx for isolated installation
  -Version   Specify version (default: latest)
  -Help      Show this help
"@
    exit 0
}

# Banner
Write-Host @"

 ██╗   ██╗██╗ ██████╗████████╗ ██████╗ ██████╗
 ██║   ██║██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
 ██║   ██║██║██║        ██║   ██║   ██║██████╔╝
 ╚██╗ ██╔╝██║██║        ██║   ██║   ██║██╔══██╗
  ╚████╔╝ ██║╚██████╗   ██║   ╚██████╔╝██║  ██║
   ╚═══╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝

          Enterprise-Ready AI Coding Assistant

"@ -ForegroundColor Blue

Write-Info "Detected: Windows $([System.Environment]::OSVersion.Version)"
Write-Host ""

# Check for Python
function Test-Python {
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 12)) {
                Write-Success "✓ Python $pythonVersion found"
                return $true
            } else {
                Write-Warning "⚠ Python $pythonVersion found, but 3.12+ required"
                return $false
            }
        }
    } catch {
        Write-Error "✗ Python 3 not found"
        return $false
    }
    return $false
}

# Install Python via winget or manual download
function Install-Python {
    Write-Warning "Installing Python..."

    # Try winget first
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install -e --id Python.Python.3.12
    } else {
        Write-Error "Please install Python 3.12+ manually from https://www.python.org/downloads/"
        exit 1
    }

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Install via pip
function Install-Pip {
    Write-Info "➤ Installing Victor via pip..."

    if ($Dev) {
        python -m pip install "victor-ai[dev]"
    } else {
        python -m pip install victor-ai
    }

    Write-Success "✓ Victor installed successfully!"
}

# Install via pipx
function Install-Pipx {
    Write-Info "➤ Installing Victor via pipx (isolated)..."

    # Check if pipx is installed
    if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
        Write-Warning "Installing pipx first..."
        python -m pip install pipx
        python -m pipx ensurepath
    }

    pipx install --python (Get-Command python).Source victor-ai

    Write-Success "✓ Victor installed successfully!"
}

# Install standalone binary
function Install-Binary {
    Write-Info "➤ Installing Victor binary..."

    $arch = if ([System.Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
    $binaryName = "victor-windows-$arch"
    $downloadUrl = "https://github.com/vijayksingh/victor/releases/latest/download/$binaryName.zip"

    $installDir = "$env:LOCALAPPDATA\Programs\Victor"
    New-Item -ItemType Directory -Force -Path $installDir | Out-Null

    Write-Info "Downloading $binaryName..."

    # Download and extract
    $zipPath = "$env:TEMP\victor.zip"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
    Remove-Item $zipPath

    # Add to PATH
    $currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$installDir*") {
        Write-Warning "Adding $installDir to PATH..."
        [System.Environment]::SetEnvironmentVariable("Path", "$currentPath;$installDir", "User")
        $env:Path = "$env:Path;$installDir"
    }

    Write-Success "✓ Victor binary installed to $installDir"
}

# Main installation
function Main {
    if ($Binary) {
        Install-Binary
    } elseif ($Pipx) {
        if (-not (Test-Python)) {
            Install-Python
            if (-not (Test-Python)) { throw "Activate Python 3.12+ and rerun the installer." }
        }
        Install-Pipx
    } else {
        if (-not (Test-Python)) {
            Install-Python
            if (-not (Test-Python)) { throw "Activate Python 3.12+ and rerun the installer." }
        }
        Install-Pip
    }

    Write-Host ""
    Write-Success "╔═══════════════════════════════════════════════════════════╗"
    Write-Success "║              Installation Complete!                       ║"
    Write-Success "╚═══════════════════════════════════════════════════════════╝"
    Write-Host ""
    Write-Host "Quick Start:"
    Write-Host ""
    Write-Host "  1. Initialize Victor in your project:"
    Write-Info "     victor init"
    Write-Host ""
    Write-Host "  2. Start chatting:"
    Write-Info "     victor chat"
    Write-Host ""
    Write-Host "  3. Or use the TUI:"
    Write-Info "     victor"
    Write-Host ""
    Write-Host "For more information, visit:"
    Write-Host "  https://github.com/vijayksingh/victor"
    Write-Host ""
}

Main
