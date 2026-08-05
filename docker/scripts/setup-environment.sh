#!/bin/bash
# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Victor Docker Setup Script
set -e

echo "===================================="
echo "Victor Docker Setup"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker from https://docker.com"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose"
    exit 1
fi

echo -e "${GREEN}✓ Docker Compose found${NC}"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env <<EOF
# Victor Docker Environment Variables

# Cloud Provider API Keys (Optional)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
XAI_API_KEY=

# Local Services (Auto-configured)
OLLAMA_HOST=http://ollama:11434
VLLM_API_BASE=http://vllm:8000/v1

# Demo Configuration
DEMO_OUTPUT_DIR=/output
EOF
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo -e "${YELLOW}Note: Edit .env to add your API keys for cloud providers${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p demo_workspace
mkdir -p notebooks
echo -e "${GREEN}✓ Directories created${NC}"

echo ""
echo "===================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "===================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Quick Start (Ollama only):"
echo "   docker-compose --profile ollama up -d"
echo ""
echo "2. Full Stack (Ollama + vLLM):"
echo "   docker-compose --profile full up -d"
echo ""
echo "3. Run Demos:"
echo "   docker-compose --profile demo up"
echo ""
echo "4. Interactive Shell:"
echo "   docker-compose run victor bash"
echo ""
echo "5. Jupyter Notebooks:"
echo "   docker-compose --profile notebook up -d"
echo "   Open: http://localhost:8888"
echo ""
echo "For more information, see docker/README.md"
echo ""
