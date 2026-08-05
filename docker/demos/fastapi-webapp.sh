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

# FastAPI Webapp Demo - Victor builds a complete web application
# This script demonstrates Victor's capabilities by having it generate
# a production-ready FastAPI application with SQLite database

set -e

echo "========================================"
echo "Victor FastAPI Webapp Demo"
echo "========================================"
echo ""
echo "This demo will use Victor to build a complete FastAPI webapp with:"
echo "  • SQLite database with SQLAlchemy models"
echo "  • REST API endpoints (CRUD operations)"
echo "  • JWT authentication"
echo "  • Pytest test suite"
echo "  • Docker deployment"
echo "  • API documentation"
echo ""
echo "Duration: ~5 minutes"
echo "Output: /output/task_manager_api/"
echo ""
read -p "Press Enter to start..."
echo ""

# Ensure output directory exists
mkdir -p /output/task_manager_api
cd /output/task_manager_api

echo "Step 1: Starting Victor with Ollama..."
echo "========================================"
echo ""

# Create the prompt file for Victor
cat > /tmp/webapp_prompts.txt << 'EOF'
I need you to build a complete production-ready FastAPI web application called "Task Manager API" in the current directory. Here are the requirements:

## Project Structure:
Create these files and directories:
- src/main.py (FastAPI application)
- src/models.py (SQLAlchemy database models)
- src/database.py (Database connection and session management)
- src/schemas.py (Pydantic request/response models)
- src/crud.py (CRUD operations)
- src/auth.py (JWT authentication)
- src/config.py (Configuration management)
- tests/test_auth.py (Authentication tests)
- tests/test_tasks.py (Task CRUD tests)
- tests/conftest.py (Pytest fixtures)
- requirements.txt (Python dependencies)
- Dockerfile (Multi-stage Docker build)
- docker-compose.yml (Service orchestration)
- README.md (Complete documentation)
- .env.example (Environment variables template)

## Database Models:
User model:
- id: Integer, primary key, autoincrement
- email: String(255), unique, indexed, not null
- hashed_password: String, not null
- full_name: String(100), nullable
- is_active: Boolean, default True
- is_admin: Boolean, default False
- created_at: DateTime, default now
- updated_at: DateTime, default now, onupdate now
- tasks: Relationship to Task model

Task model:
- id: Integer, primary key, autoincrement
- title: String(200), not null
- description: Text, nullable
- priority: Enum("low", "medium", "high"), default "medium"
- status: Enum("todo", "in_progress", "done"), default "todo"
- due_date: DateTime, nullable
- completed_at: DateTime, nullable
- user_id: Integer, ForeignKey(users.id), indexed, not null
- created_at: DateTime, default now
- updated_at: DateTime, default now, onupdate now
- owner: Relationship to User model

## API Endpoints:

### Authentication (/auth):
- POST /auth/register - Register new user (email, password, full_name)
- POST /auth/login - Login and get JWT access token
- POST /auth/refresh - Refresh access token

### Users (/users):
- GET /users/me - Get current user profile
- PUT /users/me - Update current user
- DELETE /users/me - Delete account

### Tasks (/tasks):
- POST /tasks - Create new task
- GET /tasks - List tasks with pagination, filtering (status, priority), sorting
- GET /tasks/{task_id} - Get task by ID
- PUT /tasks/{task_id} - Update task
- DELETE /tasks/{task_id} - Delete task
- GET /tasks/stats - Get task statistics (count by status, priority)

## Authentication:
- Use python-jose for JWT tokens
- Use passlib with bcrypt for password hashing
- Access token expires in 30 minutes
- Include OAuth2 password bearer authentication
- Protect all endpoints except /auth/*

## Requirements.txt should include:
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
python-dotenv>=1.0.0

## Docker Configuration:
Dockerfile:
- Use python:3.12-slim as base
- Multi-stage build (builder + runtime)
- Install dependencies in builder stage
- Copy only necessary files to runtime
- Run as non-root user (uid 1000)
- Expose port 8000
- Health check on /health endpoint
- CMD: uvicorn src.main:app --host 0.0.0.0 --port 8000

docker-compose.yml:
- webapp service (build from Dockerfile)
- Volume for SQLite database persistence
- Port mapping 8000:8000
- Environment variables from .env
- Health check configured
- Restart policy: unless-stopped

## Testing:
Write comprehensive pytest tests:
- Test user registration (valid, duplicate email, invalid email)
- Test login (valid credentials, invalid credentials, inactive user)
- Test JWT token validation
- Test task CRUD operations
- Test task ownership validation
- Test pagination and filtering
- Test unauthorized access
- Include fixtures for: test database, test client, authenticated user, sample tasks

## README.md:
Include:
- Project description
- Features list
- Tech stack
- Installation instructions (Docker and local)
- Environment variables
- API endpoints with examples
- Running tests
- Development guide
- License (Apache 2.0)

## Additional Requirements:
- Use proper HTTP status codes (200, 201, 400, 401, 403, 404, 500)
- Include request validation with Pydantic
- Add comprehensive error handling
- Include CORS middleware
- Add /health endpoint for health checks
- Add /docs endpoint with OpenAPI documentation
- Use environment variables for configuration (SECRET_KEY, DATABASE_URL, etc.)
- Add logging throughout
- Include docstrings for all functions and classes

## Code Quality:
- Follow PEP 8 style guide
- Use type hints throughout
- Add docstrings to all public functions
- Include inline comments for complex logic
- Use meaningful variable names

Please create all these files with complete, production-ready code. Make sure everything works together and is properly tested.
EOF

echo "Prompt created. Now running Victor to generate the application..."
echo ""

# Run Victor with the prompts
docker exec -i victor-app victor << EOF
$(cat /tmp/webapp_prompts.txt)
EOF

echo ""
echo "========================================"
echo "Step 2: Testing the generated application"
echo "========================================"
echo ""

# Check if files were created
if [ -f "requirements.txt" ] && [ -f "src/main.py" ] && [ -f "Dockerfile" ]; then
    echo "✅ Application files generated successfully!"
    echo ""
    echo "Generated files:"
    find . -type f -name "*.py" -o -name "*.txt" -o -name "Dockerfile" -o -name "*.yml" -o -name "*.md" | sort
    echo ""
else
    echo "❌ Error: Some files were not generated"
    exit 1
fi

echo "========================================"
echo "Step 3: Building Docker image"
echo "========================================"
echo ""

docker build -t task-manager-api:demo .

echo ""
echo "========================================"
echo "Step 4: Starting the application"
echo "========================================"
echo ""

docker-compose up -d

# Wait for app to be ready
echo "Waiting for application to start..."
sleep 10

# Test the health endpoint
echo ""
echo "Testing health endpoint..."
curl -f http://localhost:8000/health || echo "Health check failed"

echo ""
echo ""
echo "========================================"
echo "✅ Demo Complete!"
echo "========================================"
echo ""
echo "The FastAPI application is now running!"
echo ""
echo "Access the API:"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • ReDoc: http://localhost:8000/redoc"
echo "  • Health: http://localhost:8000/health"
echo ""
echo "Try it out:"
echo "  1. Open http://localhost:8000/docs in your browser"
echo "  2. Register a new user via /auth/register"
echo "  3. Login via /auth/login to get a token"
echo "  4. Use the token to create/manage tasks"
echo ""
echo "Application files: /output/task_manager_api/"
echo ""
echo "To stop the application:"
echo "  docker-compose down"
echo ""
echo "This demo showcases Victor's ability to:"
echo "  ✓ Generate complete, production-ready applications"
echo "  ✓ Create proper project structure"
echo "  ✓ Implement authentication and authorization"
echo "  ✓ Write comprehensive tests"
echo "  ✓ Create Docker deployment configuration"
echo "  ✓ Generate documentation"
echo ""
echo "All generated by AI, ready for production!"
echo "========================================"
