#!/bin/bash
# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

# Victor Docker Image Builder
# Builds the Victor AI Docker image with various configurations.
#
# Usage:
#   ./scripts/build_docker.sh [command] [options]
#
# Commands:
#   full      - Build the full Victor image (default)
#   core      - Build the minimal core image
#   mcp       - Build the MCP server image
#   all       - Build all images
#   help      - Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Docker image configuration
DOCKER_REGISTRY="vjsingh1984"
IMAGE_NAME="victor-ai"
VERSION=$(grep -m1 'version = ' "$PROJECT_ROOT/pyproject.toml" | cut -d'"' -f2 || echo "latest")

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Print header
print_header() {
    echo ""
    print_msg "$BLUE" "======================================"
    print_msg "$BLUE" "  Victor Docker Image Builder"
    print_msg "$BLUE" "======================================"
    echo ""
}

# Show help
show_help() {
    print_header
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  full      Build the full Victor image (default)"
    echo "  core      Build the minimal core image"
    echo "  mcp       Build the MCP server image"
    echo "  all       Build all images"
    echo "  help      Show this help message"
    echo ""
    echo "Options:"
    echo "  --tag TAG       Custom tag (default: version from pyproject.toml)"
    echo "  --no-cache      Build without Docker cache"
    echo "  --push          Push to Docker Hub after building"
    echo ""
    echo "Image naming convention:"
    echo "  - Full image:  ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
    echo "  - Core image:  ${DOCKER_REGISTRY}/${IMAGE_NAME}:core"
    echo "  - MCP image:   ${DOCKER_REGISTRY}/${IMAGE_NAME}:mcp-server"
    echo ""
    echo "Examples:"
    echo "  $0                         # Build full image with version tag"
    echo "  $0 full --tag latest       # Build full image with 'latest' tag"
    echo "  $0 mcp --push              # Build and push MCP server image"
    echo "  $0 all --no-cache          # Build all images without cache"
    echo ""
}

# Check Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_msg "$RED" "Error: Docker is not installed"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_msg "$RED" "Error: Docker daemon is not running"
        exit 1
    fi
}

# Build full image
build_full() {
    local tag="${1:-$VERSION}"
    local no_cache="${2:-false}"

    print_msg "$BLUE" "Building full Victor image..."
    print_msg "$YELLOW" "  Image: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
    print_msg "$YELLOW" "  Dockerfile: Dockerfile"
    echo ""

    cd "$PROJECT_ROOT"

    local build_args=("build")
    if [ "$no_cache" = "true" ]; then
        build_args+=("--no-cache")
    fi
    build_args+=("-t" "${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}")
    build_args+=("-t" "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest")
    build_args+=("-f" "Dockerfile")
    build_args+=(".")

    docker "${build_args[@]}"

    print_msg "$GREEN" "Successfully built: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
}

# Build core image
build_core() {
    local tag="${1:-core}"
    local no_cache="${2:-false}"

    print_msg "$BLUE" "Building core Victor image..."
    print_msg "$YELLOW" "  Image: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
    print_msg "$YELLOW" "  Dockerfile: docker/Dockerfile.core"
    echo ""

    cd "$PROJECT_ROOT"

    local build_args=("build")
    if [ "$no_cache" = "true" ]; then
        build_args+=("--no-cache")
    fi
    build_args+=("-t" "${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}")
    build_args+=("-f" "docker/Dockerfile.core")
    build_args+=(".")

    docker "${build_args[@]}"

    print_msg "$GREEN" "Successfully built: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
}

# Build MCP server image
build_mcp() {
    local tag="${1:-mcp-server}"
    local no_cache="${2:-false}"

    print_msg "$BLUE" "Building MCP server image..."
    print_msg "$YELLOW" "  Image: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
    print_msg "$YELLOW" "  Dockerfile: docker/mcp-server/Dockerfile"
    echo ""

    cd "$PROJECT_ROOT"

    local build_args=("build")
    if [ "$no_cache" = "true" ]; then
        build_args+=("--no-cache")
    fi
    build_args+=("-t" "${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}")
    build_args+=("-f" "docker/mcp-server/Dockerfile")
    build_args+=(".")

    docker "${build_args[@]}"

    print_msg "$GREEN" "Successfully built: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
}

# Push image to Docker Hub
push_image() {
    local image_tag="$1"

    print_msg "$BLUE" "Pushing ${image_tag} to Docker Hub..."

    # Check if logged in
    if ! docker info 2>/dev/null | grep -q "Username"; then
        print_msg "$YELLOW" "Not logged in to Docker Hub. Please log in:"
        docker login
    fi

    docker push "${image_tag}"
    print_msg "$GREEN" "Successfully pushed: ${image_tag}"
}

# Main
main() {
    local command="${1:-full}"
    shift || true

    local custom_tag=""
    local no_cache="false"
    local do_push="false"

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tag)
                custom_tag="$2"
                shift 2
                ;;
            --no-cache)
                no_cache="true"
                shift
                ;;
            --push)
                do_push="true"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    print_header
    check_docker

    case $command in
        full)
            local tag="${custom_tag:-$VERSION}"
            build_full "$tag" "$no_cache"
            if [ "$do_push" = "true" ]; then
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
            fi
            ;;
        core)
            local tag="${custom_tag:-core}"
            build_core "$tag" "$no_cache"
            if [ "$do_push" = "true" ]; then
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
            fi
            ;;
        mcp)
            local tag="${custom_tag:-mcp-server}"
            build_mcp "$tag" "$no_cache"
            if [ "$do_push" = "true" ]; then
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:${tag}"
            fi
            ;;
        all)
            build_full "${custom_tag:-$VERSION}" "$no_cache"
            build_core "core" "$no_cache"
            build_mcp "mcp-server" "$no_cache"
            if [ "$do_push" = "true" ]; then
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:${custom_tag:-$VERSION}"
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:core"
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:mcp-server"
            fi
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_msg "$RED" "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac

    echo ""
    print_msg "$GREEN" "Build completed successfully!"
    echo ""
    print_msg "$YELLOW" "To run the container:"
    echo "  docker run -it --rm ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
    echo ""
    print_msg "$YELLOW" "To push to Docker Hub:"
    echo "  docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
    echo ""
}

main "$@"
