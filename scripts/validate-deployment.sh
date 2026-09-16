#!/bin/bash
# Victor Pre-Deployment Validation Script
# Validates that everything is ready for production deployment

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Victor Pre-Deployment Validation    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════╝${NC}"
echo ""

ERRORS=0
WARNINGS=0
PASSED=0

# Function to check and report
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    PASSED=$((PASSED + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"
echo ""

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    check_pass "Docker installed (version $DOCKER_VERSION)"
else
    check_fail "Docker not installed"
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f4 | cut -d',' -f1)
    check_pass "Docker Compose installed (version $COMPOSE_VERSION)"
else
    check_fail "Docker Compose not installed"
fi

# Check Docker daemon
if docker ps &> /dev/null; then
    check_pass "Docker daemon is running"
else
    check_fail "Docker daemon is not running"
fi

# Check available disk space
AVAILABLE_GB=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G.*//')
if [ "${AVAILABLE_GB%.*}" -gt 10 ]; then
    check_pass "Sufficient disk space (${AVAILABLE_GB}GB available)"
else
    check_warn "Low disk space (${AVAILABLE_GB}GB available, recommend 10GB+)"
fi

echo ""
echo -e "${YELLOW}[2/7] Validating configuration files...${NC}"
echo ""

# Check Docker Compose files
if [ -f "docker-compose.production.yml" ]; then
    check_pass "docker-compose.production.yml exists"

    # Validate syntax
    if docker-compose -f docker-compose.production.yml config --quiet; then
        check_pass "docker-compose.production.yml is valid"
    else
        check_fail "docker-compose.production.yml has syntax errors"
    fi
else
    check_fail "docker-compose.production.yml not found"
fi

# Check Dockerfile
if [ -f "Dockerfile" ]; then
    check_pass "Dockerfile exists"
else
    check_fail "Dockerfile not found"
fi

# Check environment template
if [ -f ".env.production.example" ]; then
    check_pass ".env.production.example exists"
else
    check_fail ".env.production.example not found"
fi

# Check if .env.production exists
if [ -f ".env.production" ]; then
    check_pass ".env.production exists"

    # Check for required variables
    if grep -q "GRAFANA_ADMIN_PASSWORD" .env.production; then
        if grep -q "GRAFANA_ADMIN_PASSWORD=CHANGEME" .env.production || grep -q "GRAFANA_ADMIN_PASSWORD=$" .env.production; then
            check_warn "Grafana admin password not set (still using default/empty)"
        else
            check_pass "Grafana admin password configured"
        fi
    else
        check_warn "GRAFANA_ADMIN_PASSWORD not set in .env.production"
    fi

    # Check for at least one API key (or air-gapped mode)
    HAS_ANTHROPIC=$(grep "^ANTHROPIC_API_KEY=sk-" .env.production 2>/dev/null || echo "")
    HAS_OPENAI=$(grep "^OPENAI_API_KEY=sk-" .env.production 2>/dev/null || echo "")
    HAS_GOOGLE=$(grep "^GOOGLE_API_KEY=" .env.production | grep -v "^GOOGLE_API_KEY=$" 2>/dev/null || echo "")
    IS_AIRGAPPED=$(grep "^VICTOR_AIRGAPPED_MODE=true" .env.production 2>/dev/null || echo "")

    if [ -n "$HAS_ANTHROPIC" ] || [ -n "$HAS_OPENAI" ] || [ -n "$HAS_GOOGLE" ] || [ -n "$IS_AIRGAPPED" ]; then
        check_pass "At least one provider configured (or air-gapped mode)"
    else
        check_warn "No API keys configured and air-gapped mode not enabled"
    fi
else
    check_warn ".env.production not found (will need to create before deployment)"
fi

echo ""
echo -e "${YELLOW}[3/7] Validating monitoring configuration...${NC}"
echo ""

# Check Prometheus config
if [ -f "monitoring/prometheus/prometheus.yml" ]; then
    check_pass "Prometheus configuration exists"
else
    check_fail "monitoring/prometheus/prometheus.yml not found"
fi

# Check alert rules
if [ -f "monitoring/prometheus/alerts.yml" ]; then
    check_pass "Alert rules configured"
else
    check_fail "monitoring/prometheus/alerts.yml not found"
fi

# Check Grafana provisioning
if [ -f "monitoring/grafana/provisioning/datasources/prometheus.yml" ]; then
    check_pass "Grafana datasource configured"
else
    check_fail "Grafana datasource configuration not found"
fi

# Check Grafana dashboards
if [ -f "monitoring/grafana/dashboards/victor-overview.json" ]; then
    check_pass "Grafana dashboard exists"
else
    check_fail "Grafana dashboard not found"
fi

# Check AlertManager config
if [ -f "monitoring/alertmanager/alertmanager.yml" ]; then
    check_pass "AlertManager configuration exists"
else
    check_warn "AlertManager configuration not found (optional)"
fi

echo ""
echo -e "${YELLOW}[4/7] Validating deployment scripts...${NC}"
echo ""

# Check deployment script
if [ -f "scripts/deploy-production.sh" ]; then
    check_pass "Deployment script exists"
    if [ -x "scripts/deploy-production.sh" ]; then
        check_pass "Deployment script is executable"
    else
        check_warn "Deployment script is not executable (run: chmod +x scripts/deploy-production.sh)"
    fi
else
    check_fail "scripts/deploy-production.sh not found"
fi

# Check smoke test script
if [ -f "scripts/smoke-test.sh" ]; then
    check_pass "Smoke test script exists"
    if [ -x "scripts/smoke-test.sh" ]; then
        check_pass "Smoke test script is executable"
    else
        check_warn "Smoke test script is not executable (run: chmod +x scripts/smoke-test.sh)"
    fi
else
    check_fail "scripts/smoke-test.sh not found"
fi

# Check rollback script
if [ -f "scripts/rollback.sh" ]; then
    check_pass "Rollback script exists"
    if [ -x "scripts/rollback.sh" ]; then
        check_pass "Rollback script is executable"
    else
        check_warn "Rollback script is not executable (run: chmod +x scripts/rollback.sh)"
    fi
else
    check_fail "scripts/rollback.sh not found"
fi

echo ""
echo -e "${YELLOW}[5/7] Checking port availability...${NC}"
echo ""

# Check if ports are available
check_port() {
    PORT=$1
    NAME=$2
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -an 2>/dev/null | grep -q ":$PORT.*LISTEN"; then
        check_warn "Port $PORT ($NAME) is already in use"
    else
        check_pass "Port $PORT ($NAME) is available"
    fi
}

check_port 8765 "Victor API"
check_port 9090 "Victor Metrics"
check_port 9091 "Prometheus"
check_port 3000 "Grafana"
check_port 9093 "AlertManager"
check_port 11434 "Ollama"

echo ""
echo -e "${YELLOW}[6/7] Checking documentation...${NC}"
echo ""

# Check documentation
if [ -f "DEPLOYMENT_GUIDE.md" ]; then
    check_pass "Deployment guide exists"
else
    check_warn "DEPLOYMENT_GUIDE.md not found"
fi

if [ -f "PRODUCTION_DEPLOYMENT_READINESS.md" ]; then
    check_pass "Production readiness checklist exists"
else
    check_warn "PRODUCTION_DEPLOYMENT_READINESS.md not found"
fi

if [ -f "INFRASTRUCTURE_SETUP_COMPLETE.md" ]; then
    check_pass "Infrastructure setup documentation exists"
else
    check_warn "INFRASTRUCTURE_SETUP_COMPLETE.md not found"
fi

echo ""
echo -e "${YELLOW}[7/7] Testing Docker image build (dry-run)...${NC}"
echo ""

# Test if we can build the image (without actually building)
if docker-compose -f docker-compose.production.yml config > /dev/null 2>&1; then
    check_pass "Docker Compose configuration is valid"
else
    check_fail "Docker Compose configuration has errors"
fi

# Check if base images are available
base_image=$(sed -n 's/^ARG UBUNTU_IMAGE=//p' Dockerfile)
if [ -n "$base_image" ] && docker pull "$base_image" > /dev/null 2>&1; then
    check_pass "Pinned base Docker image is available"
else
    check_warn "Could not pull base image (may need internet connection)"
fi

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Validation Results             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASSED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "  ${RED}Errors:${NC}   $ERRORS"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  All Validations Passed! ✓            ║${NC}"
    echo -e "${GREEN}║  Ready for Production Deployment      ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Review/create .env.production (if not done)"
    echo "  2. Run: ./scripts/deploy-production.sh"
    echo "  3. Run: ./scripts/smoke-test.sh"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}╔═══════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  Validation Passed with Warnings      ║${NC}"
    echo -e "${YELLOW}║  Review warnings before deployment    ║${NC}"
    echo -e "${YELLOW}╚═══════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Recommended actions:${NC}"
    echo "  - Review warnings above"
    echo "  - Create/update .env.production if needed"
    echo "  - Set Grafana admin password"
    echo "  - Configure at least one API key (or enable air-gapped mode)"
    echo ""
    echo -e "${YELLOW}Then proceed with deployment:${NC}"
    echo "  ./scripts/deploy-production.sh"
    exit 0
else
    echo -e "${RED}╔═══════════════════════════════════════╗${NC}"
    echo -e "${RED}║  Validation Failed!                   ║${NC}"
    echo -e "${RED}║  Fix errors before deployment         ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Required actions:${NC}"
    echo "  - Fix all errors listed above"
    echo "  - Re-run: ./scripts/validate-deployment.sh"
    exit 1
fi
