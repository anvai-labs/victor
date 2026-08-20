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

"""Task type classification using semantic embeddings.

This module provides task type classification for detecting:
- CREATE_SIMPLE: Simple standalone code generation (no exploration needed)
- CREATE: Create new code that integrates with existing codebase
- EDIT: Modify existing code/files
- SEARCH: Find/locate code or files
- ANALYZE: Count, measure, analyze code metrics
- DESIGN: Conceptual/planning tasks (no tools needed)
- GENERAL: Ambiguous or general help

Uses semantic similarity with embeddings instead of hardcoded regex patterns.
This handles variations and paraphrases more robustly.

Additionally provides:
- Preprocessing: Normalizes prompts by stripping file paths and identifiers
  before embedding comparison to prevent false semantic matches
- Nudging: Rule-based adjustments for known edge cases that semantic
  similarity struggles with (e.g., "read and explain" vs "edit")

SOLID Compliance (Phase 2 Refactoring):
- Uses TaskType from victor.classification.pattern_registry (Single Source of Truth)
- Uses NudgeEngine from victor.classification.nudge_engine (DIP compliance)
- Embedding logic remains here (SRP - semantic classification only)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from victor.classification import TaskType, NudgeEngine, get_nudge_engine
from victor.storage.embeddings.collections import (
    CollectionItem,
    StaticEmbeddingCollection,
)
from victor.storage.embeddings.service import EmbeddingService, get_embedding_service

if TYPE_CHECKING:
    from victor.classification.nudge_engine import NudgeEngine as NudgeEngineType

logger = logging.getLogger(__name__)

# TaskType is now imported from victor.classification.pattern_registry
# This ensures a single source of truth across the codebase


@dataclass
class TaskTypeResult:
    """Result of task type classification."""

    task_type: TaskType
    confidence: float  # 0-1 confidence score
    top_matches: List[Tuple[str, float]]  # Top matching phrases with scores
    has_file_context: bool  # Whether the prompt mentions specific files
    nudge_applied: Optional[str] = None  # Name of nudge rule applied, if any
    preprocessed_prompt: Optional[str] = None  # Preprocessed prompt used for classification


# NudgeRule is now imported from victor.classification.nudge_engine
# NUDGE_RULES are now managed by the NudgeEngine singleton from classification module
# This eliminates ~800 lines of duplicate pattern definitions
# See victor/classification/pattern_registry.py for the unified pattern definitions


# Canonical phrases for each task type
# These are used to build embedding collections for semantic matching
# More phrases = better semantic coverage for variations

CREATE_SIMPLE_PHRASES = [
    # "simple" + code type patterns
    "create a simple function",
    "write a simple script",
    "generate a simple class",
    "make a simple function",
    "write a simple Python function",
    "create a simple Python script",
    "generate a simple utility function",
    "create a simple helper function",
    "write a simple utility",
    "make a simple Python class",
    # "function that" patterns - common standalone requests
    "create a function that calculates",
    "write a function that validates",
    "generate a function that parses",
    "make a function that converts",
    "create a function to compute",
    "write a function to handle",
    "create a function that checks",
    "write a function that returns",
    "generate a function that processes",
    "make a function that formats",
    "implement a function that sorts",
    "create a function that filters",
    # Direct generation requests
    "write me a function",
    "generate me some code",
    "create code that",
    "implement a function",
    "just create a function",
    "just write a simple",
    "can you write a function",
    "please create a function",
    "I need a function that",
    "give me code for",
    # Standalone algorithm/utility requests
    "create a factorial function",
    "write a fibonacci function",
    "generate a sorting function",
    "implement an algorithm for",
    "create a utility to calculate",
    "write a helper for",
    "create a parser for",
    "implement a validator for",
    "write a converter function",
    "create a calculator function",
    # Common programming tasks (standalone)
    "write code to calculate",
    "create a function for prime numbers",
    "implement binary search",
    "write a recursive function",
    "create a lambda function",
    "implement a decorator",
    "write a generator function",
    "create an async function",
]

CREATE_PHRASES = [
    # Create with location context
    "create a new file in",
    "write a new module in",
    "add a new class to",
    "create a new component in",
    "write a new test file for",
    "create a file called",
    "make a new file in the",
    "add a new file to",
    # Create with integration
    "create a new feature that integrates",
    "write a new handler for",
    "add a new endpoint to",
    "create a new service for",
    "implement a new tool for",
    "create a provider for",
    "add a new command to",
    # New file creation with path
    "create a new Python file",
    "write a new module called",
    "add a new config file",
    "create a new script for",
    "create victor/tools/new_tool.py",
    "write tests/test_new.py",
    "add a new test file",
    # Build/implement with context
    "build a new feature",
    "implement a new system",
    "create the infrastructure for",
    "set up a new module",
    "scaffold a new component",
    "bootstrap a new service",
    # Integration patterns
    "create a tool that uses",
    "implement integration with",
    "add support for",
    "create a wrapper for",
]

EDIT_PHRASES = [
    # Direct edit patterns - HIGH PRIORITY for "Edit <file>" commands
    "edit the file",
    "edit this file",
    "edit base.py",
    "edit orchestrator.py",
    "edit the py file",
    "edit the file to add",
    "edit the file to fix",
    "edit the file to change",
    "modify this file to",
    "update this file to",
    "change the file to",
    # Edit <path> patterns (generic paths)
    "edit tools/base.py",
    "edit agent/orchestrator.py",
    "edit src/main.py",
    "edit tests/test.py",
    "edit the module",
    "edit the script",
    "edit the test file",
    # Modify patterns
    "modify the function",
    "update the code in",
    "change the implementation",
    "fix the bug in",
    "change the file",
    "update this file",
    "modify this function",
    # Add to existing patterns
    "add a method to",
    "add error handling to",
    "add logging to the file",
    "add a property to the class",
    "add validation to",
    "add a parameter to",
    "add a new field to",
    "add tests for",
    "add documentation to",
    "add type hints to",
    "add a docstring to",
    "insert code into",
    # Remove/delete patterns
    "remove the duplicate code",
    "delete the unused function",
    "remove the commented code",
    "delete this method",
    "remove the import",
    "clean up unused variables",
    "remove dead code",
    # Refactor patterns
    "refactor the function",
    "rename the variable",
    "extract the method",
    "move the code to",
    "clean up the code",
    "refactor this class",
    "simplify the logic",
    "optimize the function",
    "restructure the code",
    "split the function into",
    # Fix patterns - many variations
    "fix the issue in",
    "fix the error in",
    "fix this code",
    "correct the bug",
    "resolve the problem in",
    "fix the failing test",
    "fix the syntax error",
    "fix the type error",
    "fix the import error",
    "debug this function",
    "patch the vulnerability",
    "fix the bug in the parser",
    "fix the bug in the function",
    "fix the bug in this file",
    "fix the problem with",
    "repair the broken function",
    "fix the issue with the",
    "resolve the bug in",
    "debug the issue in",
    "troubleshoot the error",
    # Update patterns
    "update the imports",
    "update the configuration",
    "change the behavior of",
    "update the version",
    "change the default value",
    "update the docstring",
    "modify the return type",
    "change the parameter",
    # Improve patterns
    "improve the performance of",
    "make this function faster",
    "enhance the error handling",
    "improve the readability",
]

SEARCH_PHRASES = [
    # Find/locate patterns
    "find all occurrences of",
    "locate the file",
    "search for the function",
    "where is the class",
    "which file contains",
    "find the file that",
    "locate where",
    "search for references to",
    "find usages of",
    "where is this defined",
    "where is the configuration",
    "where is configured",
    "where is the connection",
    "where is the database",
    "where is the setting",
    # Show/list patterns
    "show me all the files",
    "list all functions that",
    "show the code that",
    "list the files in",
    "show me the classes",
    "list all imports",
    "show the methods in",
    "list all tests",
    "show me the directory structure",
    "list all the endpoints",
    "list all API endpoints",
    "list all the API endpoints",
    "list the endpoints in",
    "show me all endpoints",
    "list all routes",
    # Look for patterns
    "look for the implementation",
    "find the definition of",
    "search the codebase for",
    "find where this is used",
    "look for TODO comments",
    "find all FIXME markers",
    "search for deprecated code",
    # Explore patterns
    "explore the module",
    "what files are in",
    "show me what's in",
    "what classes exist",
    "what functions are available",
    "how is this implemented",
    "where does this come from",
    "what endpoints are there",
    "what routes exist",
    # Find all/classes patterns
    "find all classes that inherit",
    "find all classes inheriting from",
    "find all subclasses of",
    "list all classes that inherit",
    "list all subclasses",
    "find classes extending",
    "what classes inherit from",
    "which classes inherit from",
    "show me all classes that",
    "get all subclasses of",
]

ANALYZE_PHRASES = [
    # Count/measure patterns
    "count the number of",
    "how many files are",
    "measure the lines of code",
    "calculate the complexity",
    "count all functions",
    "how many classes",
    "count the tests",
    "how many lines",
    # Statistics patterns
    "show statistics for",
    "analyze the code coverage",
    "what is the total",
    "how much code is",
    "give me metrics for",
    "show code statistics",
    "calculate test coverage",
    "measure the size of",
    # Analysis patterns
    "analyze the dependencies",
    "review the code quality",
    "check the test coverage",
    "assess the performance",
    "analyze the imports",
    "review the architecture",
    "check for code smells",
    "assess the complexity",
    "analyze the structure",
    # Structural/architecture analysis (abstract patterns)
    "structural analysis",
    "framework analysis",
    "architecture analysis",
    "code structure review",
    "system architecture review",
    "analyze the framework",
    "review the framework",
    "examine the architecture",
    "assess the architecture",
    "analyze system design",
    "review system structure",
    # Comparison patterns
    "compare the implementations",
    "difference between",
    "what changed in",
    "compare versions",
    # Read and explain patterns (code understanding)
    "read and explain",
    "read the code and explain",
    "analyze the code in",
    "analyze this file",
    "analyze the module",
    "review the code in",
    "examine the code in",
    "look at the code and explain",
    "read the file and tell me",
    "summarize the code in",
    "understand the code in",
]

DESIGN_PHRASES = [
    # Conceptual patterns
    "design a system for",
    "plan the architecture",
    "outline the approach",
    "describe how to implement",
    "design the structure",
    "architect a solution",
    "plan the implementation",
    "propose an approach",
    "design a rate limiting system",
    "design a caching system",
    "design authentication",
    # No-code patterns
    "without reading any code",
    "just explain the concept",
    "give me an overview",
    "conceptually describe",
    "explain without looking at code",
    "describe the theory",
    "explain the algorithm",
    "describe the pattern",
    "just list the components",
    "just outline",
    # Planning patterns
    "what would be the best way",
    "how should I approach",
    "what is the recommended",
    "explain the design for",
    "what's the best practice",
    "how would you implement",
    "suggest an approach",
    "recommend a solution",
    "what's the best way to implement",
    "how would you design",
    "what approach should I take",
    # High-level questions
    "what components do I need",
    "what modules should I create",
    "how should I structure",
    "what's the ideal architecture",
    "design a workflow for",
    "outline the steps to",
    "what are the key components for",
    "what's needed for",
    "explain the concepts behind",
    # Explain/compare conceptual patterns
    "explain the difference between",
    "what is the difference between",
    "compare lists and tuples",
    "explain lists vs tuples",
    "difference between lists and",
    "difference between arrays and",
    "explain how X works",
    "describe the difference",
    "what are the differences",
    # "What does X do" patterns (understanding)
    "what does the class do",
    "what does the function do",
    "what does the method do",
    "what does this code do",
    "what does the module do",
    "what does this module do",
    "how does the class work",
    "how does this function work",
    "how does the system work",
    "explain what this does",
    "tell me what this does",
    "describe what this does",
    "what is the purpose of",
    "what is the role of",
]

GENERAL_PHRASES = [
    # Ambiguous help requests - no specific action
    "help me with this",
    "help me with this code",
    "help with this",
    "can you help me",
    "I need help",
    "assist me with",
    "need help with",
    "could you help",
    "please help",
    "what should I do",
    "I'm stuck",
    "having trouble with",
    "not sure what to do",
    # Vague code references
    "look at this code",
    "check this code",
    "review this",
    "what do you think",
    "is this correct",
    "is this ok",
    "any suggestions",
    "what's wrong here",
    # Very short/vague
    "this code",
    "help please",
    "not working",
    "doesn't work",
    "broken",
    "something wrong",
]

# Action phrases: git operations, test runs, script execution
ACTION_PHRASES = [
    # Git commit operations
    "commit all changes",
    "commit these changes",
    "commit the changes",
    "commit my changes",
    "commit with message",
    "stage and commit",
    "git commit",
    "make a commit",
    "create a commit",
    "commit the files",
    "commit everything",
    # Git push/pull operations
    "push to remote",
    "push the changes",
    "git push",
    "pull from remote",
    "git pull",
    "sync with remote",
    "push to origin",
    "push to main",
    "push to master",
    # Git branch operations
    "create a branch",
    "switch to branch",
    "merge the branch",
    "git merge",
    "rebase onto main",
    "cherry pick commit",
    "git stash",
    "stash changes",
    # Pull request operations
    "create a pull request",
    "open a PR",
    "make a PR",
    "create PR",
    "submit pull request",
    "open pull request",
    "create a PR for",
    # Grouped commit operations
    "group commits by",
    "grouped commits",
    "separate commits for",
    "organize commits",
    "split into commits",
    "commit in groups",
    # Test execution
    "run the tests",
    "run all tests",
    "execute tests",
    "run pytest",
    "run unittest",
    "run the test suite",
    "npm test",
    "cargo test",
    "run unit tests",
    "run integration tests",
    "execute the test suite",
    "run e2e tests",
    # Script/command execution
    "run the script",
    "execute the script",
    "run this script",
    "execute this command",
    "run the command",
    "start the server",
    "run the build",
    "execute the build",
    # Build/deploy operations
    "build the project",
    "compile the code",
    "deploy the app",
    "build and deploy",
    "npm build",
    "pip install",
    "cargo build",
    "npm install",
    "install dependencies",
    # File operations
    "move all files",
    "rename the files",
    "copy these files",
    "delete old files",
    "clean up temp files",
    "remove unused files",
    # Web search and API research
    "perform websearch",
    "perform a websearch",
    "perform web search",
    "do a web search",
    "run a websearch",
    "search the web for",
    "search the internet for",
    "search online for",
    "websearch for API",
    "websearch for documentation",
    "fetch the API documentation",
    "fetch for API",
    "fetch API docs",
    "research the API",
    "investigate the library",
    "explore the package API",
    "find the API documentation",
    "look up the API",
]

# Deep analysis phrases: comprehensive codebase exploration
ANALYSIS_DEEP_PHRASES = [
    # Full codebase analysis
    "analyze the entire codebase",
    "analyze the codebase",
    "review the entire project",
    "audit the codebase",
    "analyze the whole project",
    "review the entire codebase",
    "full codebase analysis",
    "complete codebase review",
    "analyze all the code",
    "review all modules",
    # Comprehensive analysis patterns
    "comprehensive code analysis",
    "thorough code review",
    "detailed analysis of",
    "full analysis of",
    "comprehensive review",
    "thorough review of",
    "detailed code review",
    "in-depth analysis",
    "deep dive into the code",
    "extensive code review",
    # Architecture analysis
    "architecture review",
    "architecture analysis",
    "architecture overview",
    "analyze the architecture",
    "review the architecture",
    "understand the architecture",
    "map the architecture",
    "document the architecture",
    # System understanding
    "explain the entire system",
    "describe the whole codebase",
    "understand the full system",
    "map the entire codebase",
    "document the codebase",
    "document all modules",
    # Security/quality audits
    "security audit",
    "security review",
    "vulnerability scan",
    "code quality audit",
    "quality assessment",
    "technical debt analysis",
    "tech debt review",
    "performance analysis",
    "optimization review",
    # Find all patterns
    "identify all issues",
    "find all problems",
    "find every bug",
    "identify all bugs",
    "find all code smells",
    "identify technical debt",
]

# Infrastructure/DevOps phrases: Kubernetes, Terraform, Docker, cloud config
INFRASTRUCTURE_PHRASES = [
    # Kubernetes patterns
    "create a kubernetes deployment",
    "write a kubernetes manifest",
    "create a k8s deployment",
    "write k8s configuration",
    "create kubernetes service",
    "define kubernetes pod",
    "write a statefulset manifest",
    "create a configmap",
    "write kubernetes ingress",
    "create kubernetes secret",
    "configure kubernetes resources",
    "set up kubernetes cluster",
    "create k8s namespace",
    "write kubernetes yaml",
    "define kubernetes deployment",
    "create kubernetes deployment for",
    "write a kubernetes deployment configuration",
    # Terraform patterns
    "create terraform configuration",
    "write terraform module",
    "create terraform resource",
    "write terraform infrastructure",
    "configure terraform provider",
    "create terraform plan",
    "write infrastructure as code",
    "define terraform variables",
    "create terraform backend",
    "write terraform state",
    "terraform for aws",
    "terraform for azure",
    "terraform for gcp",
    # Docker patterns
    "create a dockerfile",
    "write a dockerfile",
    "optimize dockerfile",
    "create docker image",
    "build docker container",
    "configure docker compose",
    "write docker-compose file",
    "create multi-stage dockerfile",
    "optimize docker build",
    "create docker network",
    "configure docker volumes",
    # Helm patterns
    "create helm chart",
    "write helm template",
    "configure helm values",
    "deploy with helm",
    "helm chart for",
    # Cloud infrastructure
    "create aws infrastructure",
    "configure azure resources",
    "set up gcp infrastructure",
    "create cloudformation template",
    "write pulumi configuration",
    "configure cloud resources",
    "set up cloud infrastructure",
]

# CI/CD pipeline phrases: GitHub Actions, Jenkins, GitLab CI
CI_CD_PHRASES = [
    # GitHub Actions patterns
    "create github actions workflow",
    "write github action",
    "configure github actions",
    "create github workflow",
    "set up github actions",
    "github actions for",
    "github actions ci",
    "github actions cd",
    "github actions pipeline",
    # Generic CI/CD patterns
    "create ci cd pipeline",
    "configure ci pipeline",
    "set up continuous integration",
    "create deployment pipeline",
    "configure continuous deployment",
    "build and deploy pipeline",
    "create build pipeline",
    "set up automated deployment",
    "configure release pipeline",
    # Jenkins patterns
    "create jenkins pipeline",
    "write jenkinsfile",
    "configure jenkins job",
    "create jenkins build",
    "jenkins pipeline for",
    # GitLab CI patterns
    "create gitlab ci",
    "write gitlab-ci yaml",
    "configure gitlab pipeline",
    "gitlab ci for",
    # Other CI tools
    "create circleci config",
    "configure travis ci",
    "set up azure pipelines",
    "create tekton pipeline",
    "configure argo cd",
]

# Data Analysis phrases: statistical analysis, data exploration
DATA_ANALYSIS_PHRASES = [
    # Statistical analysis patterns
    "analyze the data",
    "perform statistical analysis",
    "calculate statistics for",
    "run statistical tests",
    "compute correlation",
    "analyze correlation between",
    "calculate regression",
    "perform hypothesis testing",
    "compute descriptive statistics",
    "analyze distribution",
    # Data exploration patterns
    "explore the dataset",
    "analyze the dataframe",
    "investigate data patterns",
    "examine data trends",
    "profile the data",
    "summarize the data",
    "analyze data quality",
    "check for outliers",
    "analyze missing values",
    "clean the data",
    # Pandas/NumPy patterns
    "analyze with pandas",
    "process dataframe",
    "transform the data",
    "aggregate data",
    "pivot the data",
    "group by and analyze",
    "merge datasets",
    "join dataframes",
    # Domain-specific analysis
    "analyze sales data",
    "analyze user data",
    "analyze log data",
    "analyze time series",
    "analyze financial data",
    "analyze metrics",
    "analyze performance data",
    "analyze sensor data",
]

# Visualization phrases: charts, graphs, dashboards
VISUALIZATION_PHRASES = [
    # Chart creation patterns
    "create a chart",
    "generate a graph",
    "make a plot",
    "create visualization",
    "build a dashboard",
    "design a chart",
    "create data visualization",
    "generate visualizations",
    # Specific chart types
    "create bar chart",
    "make line chart",
    "create scatter plot",
    "generate histogram",
    "create pie chart",
    "make box plot",
    "create heatmap",
    "generate treemap",
    "create area chart",
    "make bubble chart",
    # Library-specific patterns
    "matplotlib chart",
    "seaborn plot",
    "plotly visualization",
    "bokeh dashboard",
    "altair chart",
    "create with matplotlib",
    "plot with seaborn",
    "interactive plotly",
    # Dashboard patterns
    "create dashboard",
    "build analytics dashboard",
    "design data dashboard",
    "create monitoring dashboard",
    "build reporting dashboard",
    # Visualization tasks
    "visualize the data",
    "plot the results",
    "chart the trends",
    "graph the metrics",
    "display the analysis",
    "show data visually",
    "illustrate the findings",
]

# Research phrases: fact checking, literature review, competitive analysis, etc.
FACT_CHECK_PHRASES = [
    # Fact checking patterns
    "fact check this claim",
    "verify the claim",
    "validate the statement",
    "check if this is true",
    "verify the information",
    "fact check the statement",
    "is this claim accurate",
    "verify these facts",
    "check the accuracy of",
    "validate the information",
    # Source verification
    "find sources for",
    "verify with sources",
    "cross-reference this claim",
    "check multiple sources",
    "find authoritative sources",
    "verify against official sources",
]

LITERATURE_REVIEW_PHRASES = [
    # Literature review patterns
    "literature review on",
    "systematic review of",
    "review the literature",
    "survey the research",
    "review existing studies",
    "academic literature on",
    "research review on",
    "scholarly review of",
    "review the papers on",
    "survey existing work",
    # Bibliography/citation
    "find papers about",
    "search academic literature",
    "review published research",
    "synthesize the research",
    "summarize the literature",
    "bibliography on",
]

COMPETITIVE_ANALYSIS_PHRASES = [
    # Competitive analysis patterns
    "competitive analysis of",
    "compare competitors",
    "competitor analysis",
    "comparison analysis",
    "compare products",
    "compare services",
    "analyze competitors",
    "evaluate competitors",
    "competitive landscape",
    "market comparison",
    # Product/service comparison
    "compare features of",
    "compare pricing of",
    "analyze market position",
    "compare tools",
    "evaluate alternatives",
    "compare solutions",
    "analyze product differences",
    "feature comparison",
]

TREND_RESEARCH_PHRASES = [
    # Trend research patterns
    "trend research on",
    "identify trends in",
    "emerging trends",
    "market trends",
    "industry trends",
    "research trends",
    "analyze trends",
    "trend analysis",
    "find emerging patterns",
    "track industry developments",
    # Future/emerging
    "emerging technologies",
    "future developments",
    "upcoming trends",
    "new developments in",
    "what's new in",
    "latest developments",
]

TECHNICAL_RESEARCH_PHRASES = [
    # Technical research patterns
    "technical research on",
    "deep dive into",
    "technical analysis of",
    "research the technology",
    "investigate the framework",
    "explore the library",
    "technical investigation",
    "protocol research",
    "architecture research",
    "implementation research",
    # Technology exploration
    "research API for",
    "explore the documentation",
    "understand the technology",
    "technical deep dive",
    "research the stack",
    "investigate the solution",
]

# Coding vertical granular task phrases
CODE_GENERATION_PHRASES = [
    # Generate code patterns
    "generate code for",
    "generate implementation",
    "generate new class",
    "generate new module",
    "create implementation for",
    "implement new feature",
    "implement new functionality",
    "implement new system",
    "write new code for",
    "build new feature",
    "develop new component",
    "code the solution",
    "implement the solution",
    "create the implementation",
    "build the component",
    "develop the feature",
]

REFACTOR_PHRASES = [
    # Refactoring patterns
    "refactor the code",
    "refactor this function",
    "refactor the class",
    "refactor the module",
    "refactor the implementation",
    "restructure the code",
    "reorganize the code",
    "clean up the code",
    "simplify the implementation",
    "extract the method",
    "rename the variable",
    "improve code structure",
    "optimize the code",
    "reduce code duplication",
    "make code more readable",
    "apply design pattern",
]

DEBUG_PHRASES = [
    # Debugging patterns
    "debug the issue",
    "debug the error",
    "debug the problem",
    "debug this code",
    "find the bug",
    "locate the error",
    "identify the issue",
    "trace the error",
    "investigate the bug",
    "diagnose the problem",
    "troubleshoot the issue",
    "find root cause",
    "what's causing this error",
    "why is this failing",
    "fix the crash",
    "track down the bug",
]

TEST_PHRASES = [
    # Testing patterns
    "write tests for",
    "write unit tests",
    "create test cases",
    "add tests to",
    "write test for function",
    "add unit test",
    "increase test coverage",
    "improve test coverage",
    "create integration tests",
    "add test coverage",
    "test the function",
    "test the class",
    "write pytest tests",
    "create mock tests",
    "test edge cases",
    "write test suite",
]

# DevOps vertical granular task phrases
DOCKERFILE_PHRASES = [
    # Dockerfile patterns
    "create a dockerfile",
    "write a dockerfile",
    "optimize dockerfile",
    "multi-stage dockerfile",
    "improve dockerfile",
    "create docker image",
    "build docker container",
    "docker build file",
    "dockerfile for python",
    "dockerfile for node",
    "create container image",
    "containerize the application",
]

DOCKER_COMPOSE_PHRASES = [
    # Docker Compose patterns
    "create docker compose",
    "write docker-compose file",
    "configure docker compose",
    "docker compose yaml",
    "compose file for",
    "multi-container docker",
    "docker compose services",
    "compose configuration",
    "docker compose network",
    "compose volumes",
    "orchestrate containers",
    "docker compose setup",
]

KUBERNETES_PHRASES = [
    # Kubernetes patterns
    "create kubernetes manifest",
    "write k8s deployment",
    "kubernetes service config",
    "create k8s config",
    "kubernetes pod spec",
    "k8s deployment yaml",
    "create statefulset",
    "kubernetes configmap",
    "k8s secret",
    "kubernetes ingress",
    "create k8s namespace",
    "kubernetes resource",
]

TERRAFORM_PHRASES = [
    # Terraform patterns
    "create terraform module",
    "write terraform config",
    "terraform resource",
    "terraform provider",
    "infrastructure as code",
    "terraform plan",
    "terraform for aws",
    "terraform for azure",
    "terraform state",
    "terraform variables",
    "terraform backend",
    "terraform output",
]

MONITORING_PHRASES = [
    # Monitoring/observability patterns
    "set up monitoring",
    "configure alerting",
    "create prometheus config",
    "grafana dashboard",
    "set up observability",
    "monitoring setup",
    "create alerts",
    "configure metrics",
    "logging setup",
    "tracing configuration",
    "set up datadog",
    "cloudwatch monitoring",
]

# Data Analysis vertical granular task phrases
DATA_PROFILING_PHRASES = [
    # Data profiling patterns
    "profile the data",
    "data profiling",
    "profile dataset",
    "analyze data quality",
    "data quality check",
    "profile dataframe",
    "examine data structure",
    "data exploration",
    "summarize dataset",
    "describe data",
    "check data types",
    "identify missing values",
]

STATISTICAL_ANALYSIS_PHRASES = [
    # Statistical analysis patterns
    "statistical analysis",
    "perform statistical test",
    "hypothesis testing",
    "significance test",
    "t-test analysis",
    "anova analysis",
    "chi-square test",
    "statistical inference",
    "compute p-value",
    "statistical significance",
    "parametric test",
    "non-parametric test",
]

CORRELATION_ANALYSIS_PHRASES = [
    # Correlation analysis patterns
    "correlation analysis",
    "correlation matrix",
    "calculate correlation",
    "pearson correlation",
    "spearman correlation",
    "analyze relationships",
    "correlation heatmap",
    "variable correlation",
    "covariance analysis",
    "correlation between",
    "cross-correlation",
    "multicollinearity check",
]

REGRESSION_PHRASES = [
    # Regression analysis patterns
    "regression analysis",
    "linear regression",
    "logistic regression",
    "build regression model",
    "fit regression",
    "multiple regression",
    "regression coefficients",
    "predict with regression",
    "regression line",
    "least squares",
    "polynomial regression",
    "ridge regression",
]

CLUSTERING_PHRASES = [
    # Clustering analysis patterns
    "clustering analysis",
    "k-means clustering",
    "hierarchical clustering",
    "dbscan clustering",
    "cluster the data",
    "segment customers",
    "cluster analysis",
    "find clusters",
    "clustering algorithm",
    "silhouette score",
    "elbow method",
    "cluster centers",
]

TIME_SERIES_PHRASES = [
    # Time series analysis patterns
    "time series analysis",
    "analyze time series",
    "forecast time series",
    "time series forecasting",
    "seasonal decomposition",
    "trend analysis",
    "arima model",
    "temporal analysis",
    "time series data",
    "seasonality analysis",
    "autocorrelation",
    "moving average",
]

# Research vertical granular task phrase
GENERAL_QUERY_PHRASES = [
    # General query patterns
    "what is",
    "explain the concept",
    "tell me about",
    "describe the",
    "how does X work",
    "what are the benefits",
    "define the term",
    "explain how",
    "overview of",
    "introduction to",
    "summary of",
    "basics of",
]


class TaskTypeClassifier:
    """Semantic task type classifier using embeddings.

    Replaces hardcoded regex patterns with semantic similarity.
    Benefits:
    - Handles variations and paraphrases
    - More robust to prompt variations
    - Reduces false positives

    Usage:
        # Use singleton to avoid duplicate initialization
        classifier = TaskTypeClassifier.get_instance()
        classifier.initialize_sync()

        result = classifier.classify(
            "Create a simple Python function that calculates factorial"
        )
        print(result.task_type)  # TaskType.CREATE_SIMPLE
        print(result.confidence)  # 0.85
    """

    _instance: Optional["TaskTypeClassifier"] = None
    _lock = __import__("threading").Lock()

    @classmethod
    def get_instance(
        cls,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        threshold: float = 0.35,
        use_weighted_similarity: bool = True,
    ) -> "TaskTypeClassifier":
        """Get or create the singleton TaskTypeClassifier instance.

        Args:
            cache_dir: Directory for cache files (only used on first call)
            embedding_service: Shared embedding service (only used on first call)
            threshold: Minimum similarity threshold (only used on first call)
            use_weighted_similarity: Use weighted cosine similarity with key term boosting
                (only used on first call). Default: True for better task classification.

        Returns:
            The singleton TaskTypeClassifier instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        cache_dir=cache_dir,
                        embedding_service=embedding_service,
                        threshold=threshold,
                        use_weighted_similarity=use_weighted_similarity,
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
            logger.debug("Reset TaskTypeClassifier singleton")

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        threshold: float = 0.35,
        nudge_engine: Optional[NudgeEngine] = None,
        use_weighted_similarity: bool = True,
    ):
        """Initialize task type classifier.

        Args:
            cache_dir: Directory for cache files
            embedding_service: Shared embedding service
            threshold: Minimum similarity threshold for classification
            nudge_engine: Optional NudgeEngine for post-classification corrections (DIP)
            use_weighted_similarity: Use weighted cosine similarity with key term boosting
                for better task classification. Default: True.
        """
        from victor.config.settings import get_project_paths

        self.cache_dir = cache_dir or get_project_paths().global_embeddings_dir
        self.embedding_service = embedding_service or get_embedding_service()
        self.threshold = threshold
        self.use_weighted_similarity = use_weighted_similarity
        # Use injected NudgeEngine or get singleton from classification module
        self._nudge_engine = nudge_engine or get_nudge_engine()

        # Phrase lists by task type
        self._phrase_lists: Dict[TaskType, List[str]] = {
            TaskType.CREATE_SIMPLE: CREATE_SIMPLE_PHRASES,
            TaskType.CREATE: CREATE_PHRASES,
            TaskType.EDIT: EDIT_PHRASES,
            TaskType.SEARCH: SEARCH_PHRASES,
            TaskType.ANALYZE: ANALYZE_PHRASES,
            TaskType.DESIGN: DESIGN_PHRASES,
            TaskType.GENERAL: GENERAL_PHRASES,
            TaskType.ACTION: ACTION_PHRASES,
            TaskType.ANALYSIS_DEEP: ANALYSIS_DEEP_PHRASES,
            # DevOps task types (main)
            TaskType.INFRASTRUCTURE: INFRASTRUCTURE_PHRASES,
            TaskType.CI_CD: CI_CD_PHRASES,
            # Data Analysis task types (main)
            TaskType.DATA_ANALYSIS: DATA_ANALYSIS_PHRASES,
            TaskType.VISUALIZATION: VISUALIZATION_PHRASES,
            # Research task types
            TaskType.FACT_CHECK: FACT_CHECK_PHRASES,
            TaskType.LITERATURE_REVIEW: LITERATURE_REVIEW_PHRASES,
            TaskType.COMPETITIVE_ANALYSIS: COMPETITIVE_ANALYSIS_PHRASES,
            TaskType.TREND_RESEARCH: TREND_RESEARCH_PHRASES,
            TaskType.TECHNICAL_RESEARCH: TECHNICAL_RESEARCH_PHRASES,
            # Coding vertical granular task types
            TaskType.CODE_GENERATION: CODE_GENERATION_PHRASES,
            TaskType.REFACTOR: REFACTOR_PHRASES,
            TaskType.DEBUG: DEBUG_PHRASES,
            TaskType.TEST: TEST_PHRASES,
            # DevOps vertical granular task types
            TaskType.DOCKERFILE: DOCKERFILE_PHRASES,
            TaskType.DOCKER_COMPOSE: DOCKER_COMPOSE_PHRASES,
            TaskType.KUBERNETES: KUBERNETES_PHRASES,
            TaskType.TERRAFORM: TERRAFORM_PHRASES,
            TaskType.MONITORING: MONITORING_PHRASES,
            # Data Analysis vertical granular task types
            TaskType.DATA_PROFILING: DATA_PROFILING_PHRASES,
            TaskType.STATISTICAL_ANALYSIS: STATISTICAL_ANALYSIS_PHRASES,
            TaskType.CORRELATION_ANALYSIS: CORRELATION_ANALYSIS_PHRASES,
            TaskType.REGRESSION: REGRESSION_PHRASES,
            TaskType.CLUSTERING: CLUSTERING_PHRASES,
            TaskType.TIME_SERIES: TIME_SERIES_PHRASES,
            # Research vertical granular task type
            TaskType.GENERAL_QUERY: GENERAL_QUERY_PHRASES,
        }

        # Merge additional phrases from vertical capabilities (OCP extension point)
        self._merge_vertical_phrases()

        # Single unified collection for all task phrases (1 file instead of 9)
        self._collection = StaticEmbeddingCollection(
            name="task_classifier",
            cache_dir=self.cache_dir,
            embedding_service=self.embedding_service,
        )

        self._initialized = False
        self._tiered_service: Optional[Any] = None  # lazily resolved on first classify

    def _get_tiered_service(self) -> Optional[Any]:
        """Return TieredDecisionService if tiered classification is enabled."""
        from victor.agent.services.classification_backend import (
            use_tiered_classification,
        )

        if not use_tiered_classification():
            return None
        if self._tiered_service is None:
            try:
                from victor.core import get_container
                from victor.agent.services.tiered_decision_service import (
                    TieredDecisionService,
                )

                self._tiered_service = get_container().get(TieredDecisionService)
            except Exception:
                pass
        return self._tiered_service

    def _merge_vertical_phrases(self) -> None:
        """Merge additional phrases from vertical capabilities.

        Queries the CapabilityRegistry for an enhanced TaskClassifierPhraseProtocol
        and extends self._phrase_lists with any returned phrases. This allows
        verticals to contribute domain-specific task classification examples
        without modifying the core classifier.
        """
        try:
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.vertical_protocols import (
                TaskClassifierPhraseProtocol,
            )

            registry = CapabilityRegistry.get_instance()
            contributor = registry.get(TaskClassifierPhraseProtocol)
            if contributor is None or not registry.is_enhanced(TaskClassifierPhraseProtocol):
                return

            extra_phrases = contributor.get_classifier_phrases()
            for task_type_value, phrases in extra_phrases.items():
                try:
                    task_type = TaskType(task_type_value)
                except ValueError:
                    logger.warning(
                        "Unknown task type '%s' from vertical phrases, skipping",
                        task_type_value,
                    )
                    continue
                if task_type in self._phrase_lists:
                    self._phrase_lists[task_type].extend(phrases)
                else:
                    self._phrase_lists[task_type] = list(phrases)

            if extra_phrases:
                logger.info(
                    "Merged %d vertical phrases across %d task types",
                    sum(len(v) for v in extra_phrases.values()),
                    len(extra_phrases),
                )
        except Exception:
            logger.debug("Failed to merge vertical phrases", exc_info=True)

    @property
    def is_initialized(self) -> bool:
        """Check if classifier is initialized."""
        return self._initialized

    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt to normalize for better embedding matching.

        This strips or normalizes:
        - Full file paths (victor/agent/foo.py → foo.py)
        - Class/function names in PascalCase/snake_case
        - Specific identifiers that cause false semantic matches

        The goal is to preserve the ACTION words and structure while removing
        specific identifiers that can cause semantic similarity issues.

        Args:
            prompt: Original user prompt

        Returns:
            Preprocessed prompt suitable for embedding comparison
        """
        processed = prompt

        # Step 1: Extract and preserve the action prefix (first 1-3 words that matter)
        # This ensures "read and explain" stays intact
        action_patterns = [
            r"^(read\s+and\s+explain)\s+",
            r"^(explain\s+what)\s+",
            r"^(what\s+does)\s+",
            r"^(find\s+all)\s+",
            r"^(list\s+all)\s+",
            r"^(edit)\s+",
            r"^(create)\s+",
            r"^(write)\s+",
            r"^(add)\s+",
            r"^(fix)\s+",
            r"^(modify)\s+",
            r"^(update)\s+",
            r"^(remove)\s+",
            r"^(delete)\s+",
            r"^(search)\s+",
            r"^(count)\s+",
            r"^(how\s+many)\s+",
        ]

        action_prefix = ""
        for pattern in action_patterns:
            match = re.match(pattern, processed, re.IGNORECASE)
            if match:
                action_prefix = match.group(1).lower() + " "
                processed = processed[match.end() :]
                break

        # Step 2: Replace full file paths with just the filename
        # "victor/agent/orchestrator.py" → "orchestrator.py"
        processed = re.sub(
            r"[\w/\\]+/([\w.-]+\.(py|js|ts|go|rs|java|yaml|yml|json|md|txt))",
            r"\1",
            processed,
        )

        # Step 3: Replace PascalCase class names with generic "the class"
        # But preserve certain keywords
        preserve_words = {
            "tool",
            "base",
            "provider",
            "agent",
            "function",
            "method",
            "file",
            "class",
            "module",
            "endpoint",
            "api",
            "test",
        }

        def replace_pascal(match):
            word = match.group(0)
            if word.lower() in preserve_words:
                return word
            return "the class"

        # Match PascalCase words (2+ uppercase letters with lowercase between)
        processed = re.sub(
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",
            replace_pascal,
            processed,
        )

        # Step 4: Reconstruct with preserved action prefix
        result = action_prefix + processed

        # Clean up extra whitespace
        result = re.sub(r"\s+", " ", result).strip()

        logger.debug(f"Preprocessed prompt: '{prompt[:50]}...' → '{result[:50]}...'")
        return result

    def _apply_nudge_rules(
        self,
        prompt: str,
        embedding_result: TaskType,
        embedding_score: float,
        all_scores: Dict[TaskType, float],
    ) -> Tuple[TaskType, float, Optional[str]]:
        """Apply nudge rules to correct embedding classification.

        Nudge rules handle edge cases where semantic similarity fails,
        such as "read and explain" being confused with "edit".

        Delegates to NudgeEngine from victor.classification module (DIP compliance).

        Args:
            prompt: Original user prompt (not preprocessed)
            embedding_result: Task type from embedding classification
            embedding_score: Confidence score from embedding
            all_scores: Scores for all task types

        Returns:
            Tuple of (final_type, final_score, nudge_name or None)
        """
        # Delegate to NudgeEngine from classification module
        return self._nudge_engine.apply(
            prompt=prompt,
            embedding_result=embedding_result,
            embedding_confidence=embedding_score,
            scores=all_scores,
        )

    def initialize_sync(self) -> None:
        """Initialize unified task classifier collection with all phrases."""
        if self._initialized:
            return

        # Build all items with task_type in metadata
        all_items: List[CollectionItem] = []
        for task_type, phrases in self._phrase_lists.items():
            for i, phrase in enumerate(phrases):
                all_items.append(
                    CollectionItem(
                        id=f"{task_type.value}_{i}",
                        text=phrase,
                        metadata={"task_type": task_type.value},
                    )
                )

        # Initialize single unified collection
        self._collection.initialize_sync(all_items)

        self._initialized = True
        logger.info(f"TaskTypeClassifier initialized with {len(all_items)} canonical phrases")

    async def initialize(self) -> None:
        """Initialize unified task classifier collection (async version)."""
        if self._initialized:
            return

        # Build all items with task_type in metadata
        all_items: List[CollectionItem] = []
        for task_type, phrases in self._phrase_lists.items():
            for i, phrase in enumerate(phrases):
                all_items.append(
                    CollectionItem(
                        id=f"{task_type.value}_{i}",
                        text=phrase,
                        metadata={"task_type": task_type.value},
                    )
                )

        # Initialize single unified collection
        await self._collection.initialize(all_items)

        self._initialized = True
        logger.info(f"TaskTypeClassifier initialized with {len(all_items)} canonical phrases")

    def _detect_file_context(self, prompt: str) -> bool:
        """Detect if prompt mentions specific file paths.

        Args:
            prompt: User prompt

        Returns:
            True if prompt contains file path references
        """
        # Common file path patterns
        patterns = [
            r"[\w/\\.-]+\.(py|js|ts|go|rs|java|yaml|yml|json|md|txt)(?:\s|$|,)",
            r"in\s+\w+/\w+",  # "in victor/agent"
            r"to\s+\w+/\w+",  # "to victor/tools"
        ]

        prompt_lower = prompt.lower()
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                return True
        return False

    def classify_sync(self, prompt: str) -> TaskTypeResult:
        """Classify the task type of a user prompt.

        Uses a three-stage approach:
        1. Preprocess: Normalize prompt to strip file paths and identifiers
        2. Embed: Use semantic similarity with preprocessed prompt
        3. Nudge: Apply rule-based corrections for edge cases

        Args:
            prompt: User prompt text

        Returns:
            TaskTypeResult with classified task type
        """
        from victor.framework.task.direct_response import (
            classify_direct_response_prompt,
        )

        direct_response = classify_direct_response_prompt(prompt)
        if direct_response.is_direct_response:
            return TaskTypeResult(
                task_type=TaskType.GENERAL_QUERY,
                confidence=direct_response.confidence,
                top_matches=[
                    (
                        f"direct_response:{direct_response.reason}",
                        direct_response.confidence,
                    )
                ],
                has_file_context=False,
                nudge_applied="direct_response",
                preprocessed_prompt=prompt,
            )

        if not self._initialized:
            self.initialize_sync()

        # Detect file context (on original prompt)
        has_file_context = self._detect_file_context(prompt)

        # Step 1: Preprocess prompt for embedding comparison
        preprocessed = self._preprocess_prompt(prompt)

        # Step 2: Search unified collection and aggregate by task_type
        # Get top 20 results to ensure we see all task types
        # Use weighted similarity if enabled for better task classification
        results = self._collection.search_sync(
            preprocessed, top_k=20, use_weighted_similarity=self.use_weighted_similarity
        )

        all_scores: Dict[TaskType, float] = {}
        all_matches: List[Tuple[str, float]] = []

        for item, score in results:
            task_type_str = item.metadata.get("task_type", "general")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.GENERAL

            # Keep best score per task type
            if task_type not in all_scores or score > all_scores[task_type]:
                all_scores[task_type] = score

            all_matches.append((f"{task_type.value}:{item.text[:40]}", score))

        # Sort matches by score
        all_matches.sort(key=lambda x: x[1], reverse=True)

        # Find best matching task type from embeddings
        if not all_scores:
            return TaskTypeResult(
                task_type=TaskType.GENERAL,
                confidence=0.0,
                top_matches=all_matches[:5],
                has_file_context=has_file_context,
                preprocessed_prompt=preprocessed,
            )

        # Get highest scoring task type from embeddings
        embedding_type = max(all_scores.keys(), key=lambda t: all_scores[t])
        embedding_score = all_scores[embedding_type]

        # Tiered classification triage: ACCEPT/VERIFY/REJECT based on confidence band.
        # Skipped unless USE_TIERED_CLASSIFICATION is enabled.
        tiered_svc = self._get_tiered_service()
        if tiered_svc is not None:
            from victor.agent.decisions.schemas import DecisionType
            from victor.agent.services.tiered_decision_service import (
                ClassificationTriage,
            )

            triage = tiered_svc.classify_with_triage(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"query": preprocessed[:300]},
                heuristic_result=embedding_type,
                heuristic_confidence=embedding_score,
            )
            if triage.triage_outcome == ClassificationTriage.REJECT:
                final_type, final_score, nudge_name = self._apply_nudge_rules(
                    prompt, TaskType.GENERAL, triage.confidence, all_scores
                )
                return TaskTypeResult(
                    task_type=final_type,
                    confidence=triage.confidence,
                    top_matches=all_matches[:5],
                    has_file_context=has_file_context,
                    nudge_applied=nudge_name,
                    preprocessed_prompt=preprocessed,
                )
            elif triage.triage_outcome == ClassificationTriage.VERIFY and triage.result is not None:
                embedding_type = triage.result
                embedding_score = triage.confidence

        # Apply threshold check
        if embedding_score < self.threshold:
            # Even below threshold, try nudge rules (they may override)
            final_type, final_score, nudge_name = self._apply_nudge_rules(
                prompt, TaskType.GENERAL, embedding_score, all_scores
            )
            return TaskTypeResult(
                task_type=final_type,
                confidence=final_score,
                top_matches=all_matches[:5],
                has_file_context=has_file_context,
                nudge_applied=nudge_name,
                preprocessed_prompt=preprocessed,
            )

        # Step 3: Apply nudge rules on ORIGINAL prompt
        final_type, final_score, nudge_name = self._apply_nudge_rules(
            prompt, embedding_type, embedding_score, all_scores
        )

        # Special handling: CREATE_SIMPLE vs CREATE
        # If CREATE_SIMPLE has highest score but there's file context, use CREATE instead
        if final_type == TaskType.CREATE_SIMPLE and has_file_context:
            # Check if CREATE score is close enough to use instead
            create_score = all_scores.get(TaskType.CREATE, 0.0)
            if create_score >= self.threshold * 0.8:  # 80% of threshold
                final_type = TaskType.CREATE
                final_score = create_score

        return TaskTypeResult(
            task_type=final_type,
            confidence=final_score,
            top_matches=all_matches[:5],
            has_file_context=has_file_context,
            nudge_applied=nudge_name,
            preprocessed_prompt=preprocessed,
        )

    async def classify(self, prompt: str) -> TaskTypeResult:
        """Classify task type (async version)."""
        if not self._initialized:
            await self.initialize()

        # Use sync version internally (embedding search is fast)
        return self.classify_sync(prompt)

    def clear_cache(self) -> None:
        """Clear cached collections."""
        self._collection.clear_cache()
        self._initialized = False
        logger.info("TaskTypeClassifier cache cleared")
