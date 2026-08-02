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

"""Project scaffolding tool for generating project templates and boilerplate.

Features:
- Project templates (FastAPI, Flask, React, etc.)
- File structure generation
- Configuration files
- Best practices setup
- Development tooling
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# Lazy-loaded presentation adapter for icon rendering
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


# Project templates
TEMPLATES = {
    "fastapi": {
        "name": "FastAPI Application",
        "description": "Production-ready FastAPI application with best practices",
        "files": [
            ("app/__init__.py", ""),
            ("app/main.py", "fastapi_main"),
            ("app/api/__init__.py", ""),
            ("app/api/routes.py", "fastapi_routes"),
            ("app/core/__init__.py", ""),
            ("app/core/config.py", "fastapi_config"),
            ("app/models/__init__.py", ""),
            ("app/schemas/__init__.py", ""),
            ("tests/__init__.py", ""),
            ("tests/test_api.py", "fastapi_test"),
            (".env.example", "env_example"),
            ("requirements.txt", "fastapi_requirements"),
            ("pyproject.toml", "pyproject_fastapi"),
            ("README.md", "readme_fastapi"),
            (".gitignore", "gitignore_python"),
        ],
    },
    "flask": {
        "name": "Flask Application",
        "description": "Flask web application with blueprints",
        "files": [
            ("app/__init__.py", "flask_init"),
            ("app/routes.py", "flask_routes"),
            ("app/models.py", "flask_models"),
            ("app/config.py", "flask_config"),
            ("tests/__init__.py", ""),
            ("tests/test_routes.py", "flask_test"),
            ("requirements.txt", "flask_requirements"),
            ("run.py", "flask_run"),
            ("README.md", "readme_flask"),
            (".gitignore", "gitignore_python"),
        ],
    },
    "python-cli": {
        "name": "Python CLI Application",
        "description": "Python command-line tool with Click",
        "files": [
            ("cli/__init__.py", ""),
            ("cli/main.py", "cli_main"),
            ("cli/commands/__init__.py", ""),
            ("cli/commands/base.py", "cli_commands"),
            ("tests/__init__.py", ""),
            ("tests/test_cli.py", "cli_test"),
            ("setup.py", "setup_py"),
            ("requirements.txt", "cli_requirements"),
            ("README.md", "readme_cli"),
            (".gitignore", "gitignore_python"),
        ],
    },
    "react-app": {
        "name": "React Application",
        "description": "Modern React app with TypeScript",
        "files": [
            ("src/App.tsx", "react_app"),
            ("src/index.tsx", "react_index"),
            ("src/components/Header.tsx", "react_header"),
            ("public/index.html", "react_html"),
            ("package.json", "react_package"),
            ("tsconfig.json", "tsconfig"),
            (".gitignore", "gitignore_node"),
            ("README.md", "readme_react"),
        ],
    },
    "microservice": {
        "name": "Microservice",
        "description": "Microservice with Docker and CI/CD",
        "files": [
            ("src/__init__.py", ""),
            ("src/main.py", "microservice_main"),
            ("src/api/__init__.py", ""),
            ("src/api/handlers.py", "microservice_handlers"),
            ("src/config.py", "microservice_config"),
            ("tests/__init__.py", ""),
            ("tests/test_api.py", "microservice_test"),
            ("Dockerfile", "dockerfile"),
            ("docker-compose.yml", "docker_compose"),
            (".dockerignore", "dockerignore"),
            ("requirements.txt", "microservice_requirements"),
            ("README.md", "readme_microservice"),
            (".gitignore", "gitignore_python"),
        ],
    },
    "python_feature": {
        "name": "Python Feature Boilerplate",
        "description": "New Python feature with source and test files",
        "files": [
            ("features/{feature_filename}", "python_feature_source"),
            ("tests/{test_filename}", "python_feature_test"),
        ],
    },
}

# Template content
TEMPLATE_CONTENT = {
    "fastapi_main": '''"""FastAPI application entry point."""

from fastapi import FastAPI
from app.core.config import settings
from app.api import routes

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{{settings.API_V1_STR}}/openapi.json"
)

app.include_router(routes.router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {{"status": "healthy"}}
''',
    "fastapi_routes": '''"""API routes."""

from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint."""
    return {{"message": "Welcome to the API"}}

@router.get("/items/{{item_id}}")
async def read_item(item_id: int):
    """Get item by ID."""
    if item_id < 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {{"item_id": item_id, "name": f"Item {{item_id}}"}}
''',
    "fastapi_config": '''"""Application configuration."""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""

    PROJECT_NAME: str = "FastAPI Application"
    API_V1_STR: str = "/api/v1"

    class Config:
        env_file = ".env"

settings = Settings()
''',
    "fastapi_test": '''"""API tests."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {{"status": "healthy"}}

def test_root():
    """Test root endpoint."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
''',
    "fastapi_requirements": """fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-multipart>=0.0.6
pytest>=7.4.0
httpx>=0.25.0
""",
    "pyproject_fastapi": """[tool.black]
line-length = 100

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
""",
    "readme_fastapi": """# FastAPI Application

Production-ready FastAPI application.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation.
""",
    "cli_main": '''"""CLI application."""

import click

@click.group()
def cli():
    """My awesome CLI tool."""
    pass

@cli.command()
@click.option('--name', default='World', help='Name to greet')
def hello(name):
    """Say hello."""
    click.echo(f'Hello, {{name}}!')

if __name__ == '__main__':
    cli()
''',
    "dockerfile": """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "docker_compose": """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
    volumes:
      - .:/app
""",
    "gitignore_python": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Testing
.coverage
htmlcov/
.pytest_cache/
""",
    "gitignore_node": """# Dependencies
node_modules/

# Production
build/
dist/

# Misc
.DS_Store
.env.local
.env.development.local
.env.test.local
.env.production.local

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
""",
    "env_example": """# Application
PROJECT_NAME=My FastAPI App
API_V1_STR=/api/v1

# Database (if needed)
# DATABASE_URL=postgresql://user:password@localhost/dbname
""",
    "python_feature_source": '''"""Implementation for {feature_name}."""

# TODO: Implement {feature_name}
def main():
    """Main function for {feature_name}."""
    print("Hello from {feature_module}!")

if __name__ == "__main__":
    main()
''',
    "python_feature_test": '''"""Tests for {feature_name}."""

import pytest

def test_{feature_module}():
    """Test {feature_name} basic functionality."""
    assert True
''',
}


@tool(
    category="scaffolding",
    priority=Priority.LOW,  # Specialized project setup tool
    access_mode=AccessMode.WRITE,  # Creates files and directories
    danger_level=DangerLevel.LOW,  # Creates new files only
    keywords=["scaffold", "template", "project", "boilerplate", "generate"],
)
async def scaffold(
    operation: str,
    template: Optional[str] = None,
    name: Optional[str] = None,
    path: Optional[str] = None,
    content: str = "",
    force: bool = False,
    variables: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create project scaffolds: generate files, directories, and boilerplate.

    Performs scaffolding operations including creating projects from templates,
    listing available templates, adding files, and initializing git repositories.
    Consolidates all scaffolding functionality into a single interface.

    Args:
        operation: Operation to perform. Options: "create", "list", "add", "init-git", "from-template".
        template: Template name (for create operation: fastapi, flask, python-cli, react-app, microservice, python_feature).
        name: Project name (for create operation, used as directory name).
        path: File path (for add operation).
        content: File content (for add operation, default: empty string).
        force: Overwrite existing directory (for create operation, default: False).
        variables: Variables for template interpolation (for create and from-template operations).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - operation: Operation performed
        - project_dir: Project directory path (create)
        - files_created: List of created files (create)
        - file_path: Created file path (add)
        - templates: Available templates (list)
        - message: Success message
        - formatted_report: Human-readable report
        - error: Error message if failed

    Examples:
        # Create new project from template
        scaffold(operation="create", template="fastapi", name="my-api")

        # List available templates
        scaffold(operation="list")

        # Add file to project
        scaffold(operation="add", path="src/utils.py", content="# Utilities")

        # Create with variable interpolation
        scaffold(operation="create", template="python_feature", name="my-feature",
                 variables={"feature_name": "User Auth", "feature_filename": "user_auth.py",
                           "test_filename": "test_user_auth.py", "feature_module": "user_auth"})

        # Initialize git repository
        scaffold(operation="init-git")

        # Create with force overwrite
        scaffold(operation="create", template="flask", name="my-app", force=True)
    """
    if not operation:
        return {"success": False, "error": "Missing required parameter: operation"}

    # Create operation
    if operation == "create":
        if not template:
            return {
                "success": False,
                "error": "Create operation requires 'template' parameter",
            }

        if not name:
            return {
                "success": False,
                "error": "Create operation requires 'name' parameter",
            }

        if template not in TEMPLATES:
            available = ", ".join(TEMPLATES.keys())
            return {
                "success": False,
                "error": f"Unknown template: {template}. Available: {available}",
            }

        # Create project directory
        project_dir = Path(name)

        if project_dir.exists() and not force:
            return {
                "success": False,
                "error": f"Directory '{name}' already exists. Use force=True to overwrite",
            }

        project_dir.mkdir(parents=True, exist_ok=True)

        # Get template
        template_info = TEMPLATES[template]
        created_files = []

        # Prepare interpolation context
        interpolation_vars = variables or {}

        # Create all files
        for file_path, content_key in template_info["files"]:
            # Interpolate variables in file path
            try:
                interpolated_path = file_path.format(**interpolation_vars)
            except KeyError as e:
                return {
                    "success": False,
                    "error": f"Missing variable for path interpolation: {e}. Required variables: {interpolation_vars}",
                }

            full_path = project_dir / interpolated_path

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Get content and interpolate variables
            if content_key:
                file_content = TEMPLATE_CONTENT.get(content_key, "")
                # Interpolate variables in content
                try:
                    file_content = file_content.format(**interpolation_vars)
                except KeyError as e:
                    return {
                        "success": False,
                        "error": f"Missing variable for content interpolation: {e}. Required variables: {interpolation_vars}",
                    }
            else:
                file_content = ""

            # Write file
            try:
                full_path.write_text(file_content)
                created_files.append(str(interpolated_path))
            except Exception as e:
                logger.error("Failed to create %s: %s", file_path, e)

        # Determine next steps
        next_steps = [f"cd {name}"]

        if template in ["fastapi", "flask", "microservice"]:
            next_steps.extend(
                [
                    "pip install -r requirements.txt",
                    "# Edit .env.example and save as .env",
                ]
            )
        elif template == "react-app":
            next_steps.extend(["npm install", "npm start"])
        elif template == "python-cli":
            next_steps.append("pip install -e .")

        next_steps.extend(["git init", "git add .", "git commit -m 'Initial commit'"])

        # Build report
        report = []
        report.append(f"Created project: {name}")
        report.append(f"Template: {template_info['name']}")
        report.append(f"Description: {template_info['description']}")
        report.append("")
        report.append("Files created:")
        for file_path in created_files:
            report.append(f"  {_get_icon('success')} {file_path}")
        report.append("")
        report.append("Next steps:")
        for step in next_steps:
            report.append(f"  {step}")

        return {
            "success": True,
            "operation": "create",
            "project_dir": name,
            "files_created": created_files,
            "template_info": {
                "name": template_info["name"],
                "description": template_info["description"],
            },
            "next_steps": next_steps,
            "formatted_report": "\n".join(report),
        }

    # List operation
    elif operation == "list":
        templates = []

        report = []
        report.append("Available Project Templates:")
        report.append("=" * 70)
        report.append("")

        for template_id, template_info in TEMPLATES.items():
            templates.append(
                {
                    "id": template_id,
                    "name": template_info["name"],
                    "description": template_info["description"],
                    "file_count": len(template_info["files"]),
                }
            )

            report.append(f"• {template_id}")
            report.append(f"  Name: {template_info['name']}")
            report.append(f"  Description: {template_info['description']}")
            report.append(f"  Files: {len(template_info['files'])}")
            report.append("")

        report.append("Usage:")
        report.append("  scaffold(operation='create', template='fastapi', name='my-project')")

        return {
            "success": True,
            "operation": "list",
            "templates": templates,
            "count": len(templates),
            "formatted_report": "\n".join(report),
        }

    # Add operation
    elif operation == "add":
        if not path:
            return {
                "success": False,
                "error": "Add operation requires 'path' parameter",
            }

        file_path = Path(path)

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_path.write_text(content)
            return {
                "success": True,
                "operation": "add",
                "file_path": path,
                "message": f"Created file: {path}",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to create file: {e}"}

    # Init-git operation
    elif operation == "init-git":
        try:
            # Use asyncio subprocess to avoid blocking event loop
            import asyncio

            # Initialize git
            proc = await asyncio.create_subprocess_exec(
                "git",
                "init",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            # Create initial commit
            proc = await asyncio.create_subprocess_exec(
                "git",
                "add",
                ".",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            proc = await asyncio.create_subprocess_exec(
                "git",
                "commit",
                "-m",
                "Initial commit",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            return {
                "success": True,
                "operation": "init-git",
                "message": "Git repository initialized with initial commit",
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Git initialization failed: {e.stderr.decode()}",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Git not found. Please install git first.",
            }

    # From-template operation (variable interpolation for templates)
    elif operation == "from-template":
        if not template:
            return {
                "success": False,
                "error": "From-template operation requires 'template' parameter",
            }

        if not variables:
            return {
                "success": False,
                "error": "From-template operation requires 'variables' parameter",
            }

        if template not in TEMPLATES:
            available = ", ".join(TEMPLATES.keys())
            return {
                "success": False,
                "error": f"Unknown template: {template}. Available: {available}",
            }

        # Get template
        template_info = TEMPLATES[template]
        created_files = []

        # Create all files with variable interpolation
        for file_path, content_key in template_info["files"]:
            # Interpolate variables in file path
            try:
                interpolated_path = file_path.format(**variables)
            except KeyError as e:
                return {
                    "success": False,
                    "error": f"Missing variable for path interpolation: {e}. Required variables: {list(variables.keys())}",
                }

            full_path = Path(interpolated_path)

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Get content and interpolate variables
            if content_key:
                file_content = TEMPLATE_CONTENT.get(content_key, "")
                # Interpolate variables in content
                try:
                    file_content = file_content.format(**variables)
                except KeyError as e:
                    return {
                        "success": False,
                        "error": f"Missing variable for content interpolation: {e}. Required variables: {list(variables.keys())}",
                    }
            else:
                file_content = ""

            # Write file
            try:
                full_path.write_text(file_content)
                created_files.append(str(interpolated_path))
            except Exception as e:
                logger.error("Failed to create %s: %s", file_path, e)
                return {
                    "success": False,
                    "error": f"Failed to create file {interpolated_path}: {e}",
                }

        # Build report
        report = []
        report.append(f"Created files from template: {template}")
        report.append(f"Template: {template_info['name']}")
        report.append(f"Description: {template_info['description']}")
        report.append("")
        report.append("Files created:")
        for file_path in created_files:
            report.append(f"  {_get_icon('success')} {file_path}")

        return {
            "success": True,
            "operation": "from-template",
            "template": template,
            "files_created": created_files,
            "count": len(created_files),
            "message": f"Created {len(created_files)} file(s) from template '{template}'",
            "formatted_report": "\n".join(report),
        }

    # Unknown operation
    else:
        return {
            "success": False,
            "error": f"Unknown operation: {operation}. Valid operations: create, list, add, init-git, from-template",
        }
