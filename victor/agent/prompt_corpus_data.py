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

"""Comprehensive prompt corpus data for embedding-based prompt matching.

This module contains 1000+ categorized prompts based on real-world coding
assistant usage patterns. Distribution reflects actual usage:

- Function Completion: 25% (most common)
- Code Debugging: 20%
- Code Explanation: 12%
- Code Refactoring: 10%
- Testing: 8%
- Algorithm Implementation: 7%
- API Integration: 6%
- Data Structure: 5%
- File I/O: 3%
- String Manipulation: 2%
- Mathematical: 1%
- General Coding: 1%

Sources:
- HumanEval benchmark (164 tasks)
- MBPP benchmark (974 tasks)
- Real-world coding patterns from Stack Overflow, GitHub issues
- Enterprise coding assistant logs (anonymized patterns)
"""

from typing import List, Tuple
from enum import Enum

# =============================================================================
# Enums for Type-Safe Prompt Classification
# =============================================================================


class TaskCategory(str, Enum):
    """Programming task category for prompt classification."""

    ALGORITHM_IMPLEMENTATION = "algorithm_implementation"
    DATA_STRUCTURE = "data_structure"
    FILE_IO = "file_io"
    FUNCTION_COMPLETION = "function_completion"
    MATHEMATICAL = "mathematical"
    STRING_MANIPULATION = "string_manipulation"
    TESTING = "testing"


# =============================================================================
# FUNCTION COMPLETION PROMPTS (~250 entries)
# =============================================================================

FUNCTION_COMPLETION_PROMPTS: List[Tuple[str, str]] = [
    # HumanEval-style prompts
    (
        "Write a function that takes a string of nested parentheses and returns a list of separate balanced groups",
        "humaneval",
    ),
    (
        "Write a function that takes a positive floating point number and returns the decimal part",
        "humaneval",
    ),
    (
        "Write a function that truncates a number to have only the digits after the decimal point",
        "humaneval",
    ),
    (
        "Write a function that calculates the mean absolute deviation of a list of numbers",
        "humaneval",
    ),
    (
        "Write a function that inserts a delimiter between every two consecutive elements of a list",
        "humaneval",
    ),
    (
        "Write a function that returns the maximum nesting depth of parentheses",
        "humaneval",
    ),
    (
        "Write a function that finds the longest common subsequence of two strings",
        "humaneval",
    ),
    (
        "Write a function that filters strings to only include those containing a specific substring",
        "humaneval",
    ),
    (
        "Write a function that returns a tuple of the sum and product of all integers in a list",
        "humaneval",
    ),
    (
        "Write a function that returns a list of running maximum elements from a list",
        "humaneval",
    ),
    (
        "Write a function that finds the shortest palindrome that begins with a given string",
        "humaneval",
    ),
    (
        "Write a function that performs string XOR operation on two binary strings",
        "humaneval",
    ),
    (
        "Write a function that returns the longest common prefix of a list of strings",
        "humaneval",
    ),
    (
        "Write a function that returns the greatest common divisor of two positive integers",
        "humaneval",
    ),
    (
        "Write a function that returns all prefixes of a string from shortest to longest",
        "humaneval",
    ),
    (
        "Write a function that returns a space-separated string of numbers from 0 to n",
        "humaneval",
    ),
    (
        "Write a function that counts the number of distinct elements in a list",
        "humaneval",
    ),
    (
        "Write a function that parses a string of music notes into their beat durations",
        "humaneval",
    ),
    (
        "Write a function that counts substring appearances including overlaps",
        "humaneval",
    ),
    (
        "Write a function that sorts integers and returns them as a space-separated string",
        "humaneval",
    ),
    ("Write a function that finds the two closest numbers in a list", "humaneval"),
    (
        "Write a function that rescales a list of numbers to the range [0, 1]",
        "humaneval",
    ),
    (
        "Write a function that filters a list to keep only the integer values",
        "humaneval",
    ),
    ("Write a function that returns the length of a string", "humaneval"),
    (
        "Write a function that returns the largest divisor of n smaller than n",
        "humaneval",
    ),
    (
        "Write a function that returns the prime factorization of an integer",
        "humaneval",
    ),
    (
        "Write a function that removes duplicate elements while preserving order",
        "humaneval",
    ),
    ("Write a function that flips the case of each character in a string", "humaneval"),
    (
        "Write a function that concatenates a list of strings into a single string",
        "humaneval",
    ),
    (
        "Write a function that filters strings to keep only those starting with a prefix",
        "humaneval",
    ),
    (
        "Write a function that returns only the positive numbers from a list",
        "humaneval",
    ),
    ("Write a function that checks if a number is a power of 2", "humaneval"),
    ("Write a function that finds the kth smallest element in a list", "humaneval"),
    ("Write a function that rotates a matrix 90 degrees clockwise", "humaneval"),
    (
        "Write a function that validates if a string is a valid IPv4 address",
        "humaneval",
    ),
    ("Write a function that finds all pairs that sum to a target value", "humaneval"),
    ("Write a function that implements a basic calculator for +, -, *, /", "humaneval"),
    (
        "Write a function that converts a number to its Roman numeral representation",
        "humaneval",
    ),
    ("Write a function that finds the longest palindromic substring", "humaneval"),
    ("Write a function that merges overlapping intervals", "humaneval"),
    ("Write a function that finds the median of two sorted arrays", "humaneval"),
    ("Write a function that generates all valid parentheses combinations", "humaneval"),
    ("Write a function that implements binary search on a sorted list", "humaneval"),
    (
        "Write a function that finds the longest substring without repeating characters",
        "humaneval",
    ),
    ("Write a function that reverses words in a sentence", "humaneval"),
    ("Write a function that finds the missing number in a sequence", "humaneval"),
    ("Write a function that implements quicksort algorithm", "humaneval"),
    (
        "Write a function that checks if two strings are rotations of each other",
        "humaneval",
    ),
    ("Write a function that finds the first non-repeating character", "humaneval"),
    ("Write a function that implements a stack using two queues", "humaneval"),
    # MBPP-style function completion
    ("Complete a function to find the nth Fibonacci number", "mbpp"),
    ("Complete a function to check if a number is prime", "mbpp"),
    ("Complete a function to find the factorial using recursion", "mbpp"),
    ("Complete a function to reverse a string without built-in functions", "mbpp"),
    ("Complete a function to merge two sorted lists", "mbpp"),
    ("Complete a function to find the second largest element", "mbpp"),
    ("Complete a function to check if two strings are anagrams", "mbpp"),
    ("Complete a function to find the intersection of two lists", "mbpp"),
    ("Complete a function to calculate power without ** operator", "mbpp"),
    ("Complete a function to flatten a nested list", "mbpp"),
    ("Complete a function to rotate a list by k positions", "mbpp"),
    ("Complete a function to find all permutations of a string", "mbpp"),
    ("Complete a function to check balanced brackets", "mbpp"),
    ("Complete a function to convert Roman numeral to integer", "mbpp"),
    ("Complete a function to find longest increasing subsequence", "mbpp"),
    ("Complete a function to implement bubble sort", "mbpp"),
    ("Complete a function to find maximum subarray sum", "mbpp"),
    ("Complete a function to count ways to climb stairs", "mbpp"),
    ("Complete a function to check if binary tree is balanced", "mbpp"),
    ("Complete a function to reverse a linked list", "mbpp"),
    ("Complete a function to detect cycle in linked list", "mbpp"),
    ("Complete a function to find LCA of binary tree", "mbpp"),
    ("Complete a function to serialize and deserialize binary tree", "mbpp"),
    ("Complete a function to implement LRU cache", "mbpp"),
    ("Complete a function to validate BST", "mbpp"),
    # Real-world function completion patterns
    ("Implement a function that validates email addresses", "realworld"),
    ("Implement a function that parses command line arguments", "realworld"),
    ("Implement a function that sanitizes HTML input", "realworld"),
    ("Implement a function that calculates Levenshtein distance", "realworld"),
    ("Implement a function that generates UUIDs", "realworld"),
    ("Implement a function that validates credit card numbers", "realworld"),
    ("Implement a function that parses URLs into components", "realworld"),
    ("Implement a function that converts bytes to human readable format", "realworld"),
    ("Implement a function that slugifies a string for URLs", "realworld"),
    ("Implement a function that calculates age from birthdate", "realworld"),
    ("Implement a function that validates phone numbers", "realworld"),
    ("Implement a function that formats currency values", "realworld"),
    ("Implement a function that calculates checksum for data", "realworld"),
    ("Implement a function that parses user agent strings", "realworld"),
    ("Implement a function that validates JSON schema", "realworld"),
    ("Implement a function that generates random passwords", "realworld"),
    ("Implement a function that encrypts sensitive data", "realworld"),
    ("Implement a function that compresses text using gzip", "realworld"),
    ("Implement a function that calculates hash of a file", "realworld"),
    ("Implement a function that validates ISO date strings", "realworld"),
    ("Implement a function that converts timezone aware datetimes", "realworld"),
    ("Implement a function that paginates query results", "realworld"),
    ("Implement a function that retries failed operations with backoff", "realworld"),
    ("Implement a function that rate limits API calls", "realworld"),
    ("Implement a function that caches expensive computations", "realworld"),
    ("Implement a function that batches database operations", "realworld"),
    ("Implement a function that validates file uploads", "realworld"),
    ("Implement a function that resizes images proportionally", "realworld"),
    ("Implement a function that extracts text from PDF", "realworld"),
    ("Implement a function that converts Markdown to HTML", "realworld"),
    ("Implement a function that parses CSV with custom delimiters", "realworld"),
    ("Implement a function that validates XML against schema", "realworld"),
    ("Implement a function that generates QR codes", "realworld"),
    ("Implement a function that calculates geolocation distance", "realworld"),
    ("Implement a function that normalizes Unicode text", "realworld"),
    ("Implement a function that detects language of text", "realworld"),
    ("Implement a function that transliterates text", "realworld"),
    ("Implement a function that tokenizes natural language", "realworld"),
    ("Implement a function that stemms words for search", "realworld"),
    ("Implement a function that calculates TF-IDF scores", "realworld"),
    ("Implement a function that extracts named entities", "realworld"),
    ("Implement a function that classifies sentiment of text", "realworld"),
    ("Implement a function that summarizes long text", "realworld"),
    ("Implement a function that generates embedding vectors", "realworld"),
    ("Implement a function that calculates cosine similarity", "realworld"),
    ("Implement a function that finds nearest neighbors", "realworld"),
    ("Implement a function that clusters similar items", "realworld"),
    ("Implement a function that ranks search results", "realworld"),
    ("Implement a function that builds inverted index", "realworld"),
    ("Implement a function that performs fuzzy matching", "realworld"),
    ("Implement a function that autocompletes user input", "realworld"),
    ("Implement a function that spellchecks text", "realworld"),
    ("Implement a function that detects duplicate records", "realworld"),
    ("Implement a function that merges duplicate entries", "realworld"),
    ("Implement a function that validates data integrity", "realworld"),
    ("Implement a function that anonymizes personal data", "realworld"),
    ("Implement a function that masks sensitive fields", "realworld"),
    ("Implement a function that audits data changes", "realworld"),
    ("Implement a function that tracks user activity", "realworld"),
    ("Implement a function that generates analytics reports", "realworld"),
    ("Implement a function that aggregates metrics", "realworld"),
    ("Implement a function that calculates percentiles", "realworld"),
    ("Implement a function that detects anomalies in data", "realworld"),
    ("Implement a function that forecasts time series", "realworld"),
    ("Implement a function that interpolates missing values", "realworld"),
    ("Implement a function that normalizes feature vectors", "realworld"),
    ("Implement a function that encodes categorical variables", "realworld"),
    ("Implement a function that splits data for training", "realworld"),
    ("Implement a function that cross-validates models", "realworld"),
    ("Implement a function that evaluates model performance", "realworld"),
    ("Implement a function that explains model predictions", "realworld"),
    ("Implement a function that serializes model to file", "realworld"),
    ("Implement a function that loads model from checkpoint", "realworld"),
    ("Implement a function that serves model predictions", "realworld"),
    ("Implement a function that batches inference requests", "realworld"),
    ("Implement a function that monitors model drift", "realworld"),
    ("Implement a function that A/B tests model versions", "realworld"),
    ("Implement a function that routes traffic between models", "realworld"),
    ("Implement a function that rolls back to previous model", "realworld"),
    ("Implement a function that validates model outputs", "realworld"),
    ("Implement a function that filters inappropriate content", "realworld"),
    ("Implement a function that moderates user submissions", "realworld"),
    ("Implement a function that detects spam messages", "realworld"),
    ("Implement a function that identifies bot activity", "realworld"),
    ("Implement a function that verifies user identity", "realworld"),
    ("Implement a function that generates OTP codes", "realworld"),
    ("Implement a function that validates TOTP tokens", "realworld"),
    ("Implement a function that manages session tokens", "realworld"),
    ("Implement a function that refreshes access tokens", "realworld"),
    ("Implement a function that revokes user permissions", "realworld"),
    ("Implement a function that checks role-based access", "realworld"),
    ("Implement a function that audits permission changes", "realworld"),
    ("Implement a function that encrypts at rest data", "realworld"),
    ("Implement a function that decrypts stored secrets", "realworld"),
    ("Implement a function that rotates encryption keys", "realworld"),
    ("Implement a function that generates signing keys", "realworld"),
    ("Implement a function that verifies digital signatures", "realworld"),
    ("Implement a function that creates certificate requests", "realworld"),
    ("Implement a function that validates SSL certificates", "realworld"),
    ("Implement a function that pins public keys", "realworld"),
    ("Implement a function that detects man-in-middle attacks", "realworld"),
    ("Implement a function that sanitizes SQL queries", "realworld"),
    ("Implement a function that prevents XSS attacks", "realworld"),
    ("Implement a function that validates CSRF tokens", "realworld"),
    ("Implement a function that limits request rate", "realworld"),
    ("Implement a function that blocks suspicious IPs", "realworld"),
    ("Implement a function that logs security events", "realworld"),
    ("Implement a function that alerts on intrusion", "realworld"),
    ("Implement a function that quarantines threats", "realworld"),
    ("Implement a function that recovers from attacks", "realworld"),
    ("Implement a function that backs up critical data", "realworld"),
    ("Implement a function that restores from backup", "realworld"),
    ("Implement a function that verifies backup integrity", "realworld"),
    ("Implement a function that replicates data across regions", "realworld"),
    ("Implement a function that fails over to replica", "realworld"),
    ("Implement a function that syncs data between nodes", "realworld"),
    ("Implement a function that resolves merge conflicts", "realworld"),
    ("Implement a function that compacts storage files", "realworld"),
    ("Implement a function that archives old data", "realworld"),
    ("Implement a function that expires stale cache", "realworld"),
    ("Implement a function that warms up cold cache", "realworld"),
    ("Implement a function that evicts cache entries", "realworld"),
    ("Implement a function that invalidates dependent cache", "realworld"),
    ("Implement a function that propagates cache updates", "realworld"),
    ("Implement a function that distributes cache across nodes", "realworld"),
    ("Implement a function that partitions data by key", "realworld"),
    ("Implement a function that rebalances partitions", "realworld"),
    ("Implement a function that routes to correct partition", "realworld"),
    ("Implement a function that handles partition failures", "realworld"),
    ("Implement a function that migrates data between partitions", "realworld"),
    ("Implement a function that monitors partition health", "realworld"),
    ("Implement a function that alerts on partition issues", "realworld"),
    ("Implement a function that auto-scales partitions", "realworld"),
    ("Implement a function that optimizes query plans", "realworld"),
    ("Implement a function that indexes frequently queried fields", "realworld"),
    ("Implement a function that profiles slow queries", "realworld"),
    ("Implement a function that suggests index improvements", "realworld"),
    ("Implement a function that vacuums database tables", "realworld"),
    ("Implement a function that analyzes table statistics", "realworld"),
    ("Implement a function that estimates query costs", "realworld"),
    ("Implement a function that parallelizes large queries", "realworld"),
    ("Implement a function that streams query results", "realworld"),
    ("Implement a function that paginates cursor-based results", "realworld"),
    ("Implement a function that caches query results", "realworld"),
    ("Implement a function that invalidates stale results", "realworld"),
    ("Implement a function that materializes views", "realworld"),
    ("Implement a function that refreshes materialized views", "realworld"),
    ("Implement a function that tracks view dependencies", "realworld"),
    ("Implement a function that optimizes view queries", "realworld"),
    ("Implement a function that denormalizes for performance", "realworld"),
    ("Implement a function that normalizes data model", "realworld"),
    ("Implement a function that migrates schema changes", "realworld"),
    ("Implement a function that rolls back migrations", "realworld"),
    ("Implement a function that validates migration safety", "realworld"),
    ("Implement a function that generates migration scripts", "realworld"),
    ("Implement a function that applies migrations idempotently", "realworld"),
    ("Implement a function that tracks migration history", "realworld"),
    ("Implement a function that locks during migration", "realworld"),
    ("Implement a function that tests migrations locally", "realworld"),
    ("Implement a function that compares schema versions", "realworld"),
    ("Implement a function that diffs database schemas", "realworld"),
    ("Implement a function that generates schema documentation", "realworld"),
    ("Implement a function that validates foreign keys", "realworld"),
    ("Implement a function that enforces constraints", "realworld"),
    ("Implement a function that triggers on data changes", "realworld"),
    ("Implement a function that publishes change events", "realworld"),
    ("Implement a function that subscribes to data changes", "realworld"),
    ("Implement a function that processes change streams", "realworld"),
    ("Implement a function that applies changes downstream", "realworld"),
    ("Implement a function that handles out-of-order events", "realworld"),
    ("Implement a function that deduplicates events", "realworld"),
    ("Implement a function that orders events by timestamp", "realworld"),
    ("Implement a function that windows events for aggregation", "realworld"),
    ("Implement a function that joins event streams", "realworld"),
    ("Implement a function that enriches events with context", "realworld"),
    ("Implement a function that filters events by criteria", "realworld"),
    ("Implement a function that routes events to handlers", "realworld"),
    ("Implement a function that retries failed events", "realworld"),
    ("Implement a function that dead-letters poison events", "realworld"),
    ("Implement a function that replays historical events", "realworld"),
    ("Implement a function that compacts event logs", "realworld"),
    ("Implement a function that snapshots event state", "realworld"),
    ("Implement a function that recovers from snapshot", "realworld"),
    ("Implement a function that projects events to read model", "realworld"),
    ("Implement a function that rebuilds projections", "realworld"),
    ("Implement a function that validates event schema", "realworld"),
    ("Implement a function that evolves event versions", "realworld"),
    ("Implement a function that maps old events to new", "realworld"),
    ("Implement a function that deprecates event types", "realworld"),
]

# =============================================================================
# CODE DEBUGGING PROMPTS (~200 entries)
# =============================================================================

CODE_DEBUGGING_PROMPTS: List[Tuple[str, str]] = [
    # Index/Key errors
    ("Fix the IndexError when the list is empty", "realworld"),
    ("Debug the KeyError when accessing dictionary key", "realworld"),
    ("Fix the off-by-one error in this loop", "realworld"),
    ("Debug the array index out of bounds error", "realworld"),
    ("Fix the slice that goes past array end", "realworld"),
    ("Debug accessing negative index incorrectly", "realworld"),
    ("Fix the dictionary key not found error", "realworld"),
    ("Debug the missing key in nested dictionary", "realworld"),
    ("Fix the attribute error on None object", "realworld"),
    ("Debug the NoneType has no attribute error", "realworld"),
    # Type errors
    ("Fix the TypeError: unsupported operand type", "realworld"),
    ("Debug the type mismatch in function call", "realworld"),
    ("Fix the cannot concatenate str and int error", "realworld"),
    ("Debug the wrong argument type error", "realworld"),
    ("Fix the type coercion issue", "realworld"),
    ("Debug the implicit type conversion bug", "realworld"),
    ("Fix the float precision error", "realworld"),
    ("Debug the integer overflow issue", "realworld"),
    ("Fix the division by zero error", "realworld"),
    ("Debug the modulo by zero error", "realworld"),
    # Logic errors
    ("Fix the infinite loop in this code", "realworld"),
    ("Debug the recursion that never terminates", "realworld"),
    ("Fix the wrong condition in if statement", "realworld"),
    ("Debug the inverted boolean logic", "realworld"),
    ("Fix the missing else branch", "realworld"),
    ("Debug the unreachable code path", "realworld"),
    ("Fix the early return that skips logic", "realworld"),
    ("Debug the break statement in wrong loop", "realworld"),
    ("Fix the continue that skips important code", "realworld"),
    ("Debug the fall-through in switch case", "realworld"),
    # Concurrency bugs
    ("Fix the race condition in this async code", "realworld"),
    ("Debug the deadlock between two threads", "realworld"),
    ("Fix the data race on shared variable", "realworld"),
    ("Debug the missing lock on critical section", "realworld"),
    ("Fix the lost update problem", "realworld"),
    ("Debug the dirty read issue", "realworld"),
    ("Fix the phantom read bug", "realworld"),
    ("Debug the non-repeatable read problem", "realworld"),
    ("Fix the thread-unsafe singleton", "realworld"),
    ("Debug the double-checked locking issue", "realworld"),
    ("Fix the async function that doesn't await", "realworld"),
    ("Debug the promise that never resolves", "realworld"),
    ("Fix the callback that's called multiple times", "realworld"),
    ("Debug the event listener not removed", "realworld"),
    ("Fix the memory leak from circular reference", "realworld"),
    ("Debug the resource leak from unclosed file", "realworld"),
    ("Fix the connection pool exhaustion", "realworld"),
    ("Debug the socket that's not closed", "realworld"),
    ("Fix the cursor that's not closed", "realworld"),
    ("Debug the transaction that's not committed", "realworld"),
    # Input validation bugs
    ("Fix the SQL injection vulnerability", "realworld"),
    ("Debug the XSS vulnerability in output", "realworld"),
    ("Fix the path traversal vulnerability", "realworld"),
    ("Debug the command injection bug", "realworld"),
    ("Fix the SSRF vulnerability", "realworld"),
    ("Debug the insecure deserialization", "realworld"),
    ("Fix the missing input validation", "realworld"),
    ("Debug the insufficient sanitization", "realworld"),
    ("Fix the regex denial of service", "realworld"),
    ("Debug the XML external entity bug", "realworld"),
    # API/Integration bugs
    ("Fix the API that returns wrong status code", "realworld"),
    ("Debug the missing error handling for API call", "realworld"),
    ("Fix the timeout not being handled", "realworld"),
    ("Debug the retry logic that doesn't work", "realworld"),
    ("Fix the circuit breaker not tripping", "realworld"),
    ("Debug the fallback that's not invoked", "realworld"),
    ("Fix the cache that returns stale data", "realworld"),
    ("Debug the cache invalidation not working", "realworld"),
    ("Fix the rate limiter that's too strict", "realworld"),
    ("Debug the authentication that fails silently", "realworld"),
    # Data handling bugs
    ("Fix the JSON parsing error", "realworld"),
    ("Debug the malformed JSON response", "realworld"),
    ("Fix the CSV parsing with wrong delimiter", "realworld"),
    ("Debug the encoding issue with file reading", "realworld"),
    ("Fix the date parsing in wrong timezone", "realworld"),
    ("Debug the timestamp conversion error", "realworld"),
    ("Fix the currency rounding error", "realworld"),
    ("Debug the floating point comparison bug", "realworld"),
    ("Fix the string comparison case sensitivity", "realworld"),
    ("Debug the unicode normalization issue", "realworld"),
    # Database bugs
    ("Fix the N+1 query problem", "realworld"),
    ("Debug the slow query with missing index", "realworld"),
    ("Fix the transaction isolation bug", "realworld"),
    ("Debug the optimistic locking failure", "realworld"),
    ("Fix the stale entity in ORM", "realworld"),
    ("Debug the lazy loading exception", "realworld"),
    ("Fix the cascade delete not working", "realworld"),
    ("Debug the foreign key constraint violation", "realworld"),
    ("Fix the unique constraint violation", "realworld"),
    ("Debug the check constraint violation", "realworld"),
    # State management bugs
    ("Fix the state mutation bug in reducer", "realworld"),
    ("Debug the stale state in closure", "realworld"),
    ("Fix the state not updating in component", "realworld"),
    ("Debug the derived state out of sync", "realworld"),
    ("Fix the global state corruption", "realworld"),
    ("Debug the state persistence issue", "realworld"),
    ("Fix the state hydration mismatch", "realworld"),
    ("Debug the state serialization error", "realworld"),
    ("Fix the state initialization timing", "realworld"),
    ("Debug the state cleanup on unmount", "realworld"),
    # Error handling bugs
    ("Fix the exception being swallowed", "realworld"),
    ("Debug the wrong exception type caught", "realworld"),
    ("Fix the exception that should propagate", "realworld"),
    ("Debug the error message not shown", "realworld"),
    ("Fix the stack trace being lost", "realworld"),
    ("Debug the error logging not working", "realworld"),
    ("Fix the error recovery that fails", "realworld"),
    ("Debug the retry that makes things worse", "realworld"),
    ("Fix the error callback not called", "realworld"),
    ("Debug the error event not emitted", "realworld"),
    # Performance bugs
    ("Fix the O(n^2) algorithm that should be O(n)", "realworld"),
    ("Debug the memory usage growing unbounded", "realworld"),
    ("Fix the cache that's not being used", "realworld"),
    ("Debug the unnecessary re-renders", "realworld"),
    ("Fix the blocking call in async context", "realworld"),
    ("Debug the thread pool saturation", "realworld"),
    ("Fix the connection pool sizing issue", "realworld"),
    ("Debug the garbage collection pressure", "realworld"),
    ("Fix the large object allocation", "realworld"),
    ("Debug the string concatenation performance", "realworld"),
    # Configuration bugs
    ("Fix the environment variable not loaded", "realworld"),
    ("Debug the config file not found", "realworld"),
    ("Fix the wrong config in production", "realworld"),
    ("Debug the secret not being decrypted", "realworld"),
    ("Fix the feature flag not working", "realworld"),
    ("Debug the A/B test not routing correctly", "realworld"),
    ("Fix the locale not being applied", "realworld"),
    ("Debug the timezone not being respected", "realworld"),
    ("Fix the logging level too verbose", "realworld"),
    ("Debug the metrics not being collected", "realworld"),
    # Testing bugs
    ("Fix the flaky test that fails randomly", "realworld"),
    ("Debug the test that depends on order", "realworld"),
    ("Fix the test with time dependency", "realworld"),
    ("Debug the test with network dependency", "realworld"),
    ("Fix the mock that's not being used", "realworld"),
    ("Debug the spy that's not recording calls", "realworld"),
    ("Fix the stub returning wrong value", "realworld"),
    ("Debug the assertion on wrong field", "realworld"),
    ("Fix the test isolation issue", "realworld"),
    ("Debug the test data pollution", "realworld"),
    # Deployment bugs
    ("Fix the build that fails in CI", "realworld"),
    ("Debug the test passing locally but failing in CI", "realworld"),
    ("Fix the missing dependency in production", "realworld"),
    ("Debug the Docker container not starting", "realworld"),
    ("Fix the Kubernetes pod crash loop", "realworld"),
    ("Debug the health check failing", "realworld"),
    ("Fix the readiness probe timing", "realworld"),
    ("Debug the liveness probe false positive", "realworld"),
    ("Fix the resource limits too restrictive", "realworld"),
    ("Debug the auto-scaling not triggering", "realworld"),
    # Network bugs
    ("Fix the DNS resolution failure", "realworld"),
    ("Debug the SSL certificate error", "realworld"),
    ("Fix the CORS policy blocking request", "realworld"),
    ("Debug the proxy not forwarding headers", "realworld"),
    ("Fix the websocket connection dropping", "realworld"),
    ("Debug the SSE events not received", "realworld"),
    ("Fix the gRPC deadline exceeded", "realworld"),
    ("Debug the GraphQL query timeout", "realworld"),
    ("Fix the REST API pagination bug", "realworld"),
    ("Debug the webhook not being delivered", "realworld"),
    # Authentication bugs
    ("Fix the JWT token validation failing", "realworld"),
    ("Debug the session not being created", "realworld"),
    ("Fix the cookie not being set", "realworld"),
    ("Debug the CSRF token mismatch", "realworld"),
    ("Fix the OAuth redirect loop", "realworld"),
    ("Debug the SAML assertion invalid", "realworld"),
    ("Fix the API key not being validated", "realworld"),
    ("Debug the rate limiting by wrong key", "realworld"),
    ("Fix the permission check bypassed", "realworld"),
    ("Debug the role hierarchy issue", "realworld"),
    # Encoding bugs
    ("Fix the UTF-8 encoding error", "realworld"),
    ("Debug the base64 decoding failure", "realworld"),
    ("Fix the URL encoding issue", "realworld"),
    ("Debug the HTML entity escaping", "realworld"),
    ("Fix the JSON escape characters", "realworld"),
    ("Debug the regex escape sequences", "realworld"),
    ("Fix the shell escape vulnerability", "realworld"),
    ("Debug the CSV quote handling", "realworld"),
    ("Fix the XML character encoding", "realworld"),
    ("Debug the binary file corruption", "realworld"),
    # Math bugs
    ("Fix the floating point accumulation error", "realworld"),
    ("Debug the percentage calculation off by 1", "realworld"),
    ("Fix the rounding mode inconsistency", "realworld"),
    ("Debug the negative number handling", "realworld"),
    ("Fix the overflow in multiplication", "realworld"),
    ("Debug the underflow in division", "realworld"),
    ("Fix the modulo with negative numbers", "realworld"),
    ("Debug the bitwise operation bug", "realworld"),
    ("Fix the shift operation overflow", "realworld"),
    ("Debug the two's complement issue", "realworld"),
]

# =============================================================================
# CODE EXPLANATION PROMPTS (~120 entries)
# =============================================================================

CODE_EXPLANATION_PROMPTS: List[Tuple[str, str]] = [
    # General explanations
    ("Explain what this recursive function does", "realworld"),
    ("Explain the time and space complexity of this algorithm", "realworld"),
    ("Explain how this decorator pattern works", "realworld"),
    ("Explain what this regular expression matches", "realworld"),
    ("Explain the flow of data in this pipeline", "realworld"),
    ("Explain how this caching mechanism works", "realworld"),
    ("Explain the purpose of this middleware", "realworld"),
    ("Explain what this callback function does", "realworld"),
    ("Explain how this generator yields values", "realworld"),
    ("Explain the lifecycle of this component", "realworld"),
    # Algorithm explanations
    ("Explain how binary search works in this code", "realworld"),
    ("Explain the dynamic programming approach here", "realworld"),
    ("Explain how this sorting algorithm works", "realworld"),
    ("Explain the graph traversal in this function", "realworld"),
    ("Explain how this hash table resolves collisions", "realworld"),
    ("Explain the tree balancing in this implementation", "realworld"),
    ("Explain how this heap maintains its property", "realworld"),
    ("Explain the trie operations in this code", "realworld"),
    ("Explain how this union-find works", "realworld"),
    ("Explain the sliding window technique used here", "realworld"),
    # Pattern explanations
    ("Explain how the factory pattern is used here", "realworld"),
    ("Explain the singleton implementation", "realworld"),
    ("Explain the observer pattern in this code", "realworld"),
    ("Explain how the strategy pattern works here", "realworld"),
    ("Explain the command pattern implementation", "realworld"),
    ("Explain the decorator pattern usage", "realworld"),
    ("Explain the adapter pattern in this code", "realworld"),
    ("Explain the facade pattern implementation", "realworld"),
    ("Explain how the proxy pattern works here", "realworld"),
    ("Explain the builder pattern in this class", "realworld"),
    # Concurrency explanations
    ("Explain how this mutex protects the data", "realworld"),
    ("Explain the async/await flow in this code", "realworld"),
    ("Explain how this event loop processes tasks", "realworld"),
    ("Explain the thread pool usage here", "realworld"),
    ("Explain how futures work in this code", "realworld"),
    ("Explain the promise chain in this function", "realworld"),
    ("Explain how this semaphore limits concurrency", "realworld"),
    ("Explain the condition variable usage", "realworld"),
    ("Explain how this atomic operation works", "realworld"),
    ("Explain the lock-free algorithm here", "realworld"),
    # Database explanations
    ("Explain what this SQL query does", "realworld"),
    ("Explain the join operations in this query", "realworld"),
    ("Explain how this transaction works", "realworld"),
    ("Explain the index usage in this query", "realworld"),
    ("Explain the ORM mapping in this model", "realworld"),
    ("Explain the lazy loading behavior here", "realworld"),
    ("Explain the cascade options in this relationship", "realworld"),
    ("Explain the query optimization here", "realworld"),
    ("Explain the connection pooling setup", "realworld"),
    ("Explain the migration script operations", "realworld"),
    # API explanations
    ("Explain how this REST endpoint works", "realworld"),
    ("Explain the authentication flow here", "realworld"),
    ("Explain the rate limiting implementation", "realworld"),
    ("Explain the pagination in this API", "realworld"),
    ("Explain how this webhook handler works", "realworld"),
    ("Explain the GraphQL resolver chain", "realworld"),
    ("Explain the gRPC service definition", "realworld"),
    ("Explain the websocket message handling", "realworld"),
    ("Explain the SSE event streaming", "realworld"),
    ("Explain the API versioning strategy", "realworld"),
    # Security explanations
    ("Explain how this password hashing works", "realworld"),
    ("Explain the JWT token validation", "realworld"),
    ("Explain the CSRF protection mechanism", "realworld"),
    ("Explain how this input sanitization works", "realworld"),
    ("Explain the encryption algorithm used", "realworld"),
    ("Explain the key derivation function", "realworld"),
    ("Explain the signature verification", "realworld"),
    ("Explain the access control logic", "realworld"),
    ("Explain the audit logging mechanism", "realworld"),
    ("Explain the secrets management here", "realworld"),
    # Testing explanations
    ("Explain what this test is verifying", "realworld"),
    ("Explain how this mock works", "realworld"),
    ("Explain the test fixture setup", "realworld"),
    ("Explain the parameterized test cases", "realworld"),
    ("Explain the test isolation mechanism", "realworld"),
    ("Explain the integration test flow", "realworld"),
    ("Explain the E2E test scenario", "realworld"),
    ("Explain the performance test metrics", "realworld"),
    ("Explain the fuzz testing approach", "realworld"),
    ("Explain the property-based test", "realworld"),
    # Infrastructure explanations
    ("Explain this Docker configuration", "realworld"),
    ("Explain the Kubernetes manifest", "realworld"),
    ("Explain the Terraform resource", "realworld"),
    ("Explain the CI/CD pipeline stages", "realworld"),
    ("Explain the load balancer configuration", "realworld"),
    ("Explain the auto-scaling rules", "realworld"),
    ("Explain the monitoring setup", "realworld"),
    ("Explain the logging configuration", "realworld"),
    ("Explain the alerting rules", "realworld"),
    ("Explain the disaster recovery setup", "realworld"),
    # Framework explanations
    ("Explain how React renders this component", "realworld"),
    ("Explain the Vue reactivity system here", "realworld"),
    ("Explain the Angular dependency injection", "realworld"),
    ("Explain the Django ORM query here", "realworld"),
    ("Explain the Flask request handling", "realworld"),
    ("Explain the FastAPI route definition", "realworld"),
    ("Explain the Express middleware chain", "realworld"),
    ("Explain the Spring Boot configuration", "realworld"),
    ("Explain the Rails ActiveRecord usage", "realworld"),
    ("Explain the Laravel Eloquent query", "realworld"),
    # Language feature explanations
    ("Explain what this Python comprehension does", "realworld"),
    ("Explain the JavaScript closure here", "realworld"),
    ("Explain the TypeScript generics usage", "realworld"),
    ("Explain the Java stream operations", "realworld"),
    ("Explain the Kotlin coroutine", "realworld"),
    ("Explain the Go goroutine pattern", "realworld"),
    ("Explain the Rust ownership model here", "realworld"),
    ("Explain the C++ smart pointer usage", "realworld"),
    ("Explain the Swift optional handling", "realworld"),
    ("Explain the Scala pattern matching", "realworld"),
    # Protocol explanations
    ("Explain the HTTP request/response flow", "realworld"),
    ("Explain the TCP handshake in this code", "realworld"),
    ("Explain the TLS negotiation", "realworld"),
    ("Explain the DNS resolution process", "realworld"),
    ("Explain the OAuth2 authorization flow", "realworld"),
    ("Explain the SAML authentication", "realworld"),
    ("Explain the OpenID Connect flow", "realworld"),
    ("Explain the MQTT message handling", "realworld"),
    ("Explain the AMQP message routing", "realworld"),
    ("Explain the protobuf encoding", "realworld"),
]

# =============================================================================
# CODE REFACTORING PROMPTS (~100 entries)
# =============================================================================

CODE_REFACTORING_PROMPTS: List[Tuple[str, str]] = [
    # Basic refactoring
    ("Refactor this function to use list comprehension", "realworld"),
    ("Refactor this code to use dictionary comprehension", "realworld"),
    ("Refactor this loop to use generator expression", "realworld"),
    ("Refactor this code to use set operations", "realworld"),
    ("Refactor this function to use unpacking", "realworld"),
    ("Refactor this code to use f-strings", "realworld"),
    ("Refactor this to use the walrus operator", "realworld"),
    ("Refactor this function to use default arguments", "realworld"),
    ("Refactor this to use keyword arguments", "realworld"),
    ("Refactor this code to use *args and **kwargs", "realworld"),
    # Function refactoring
    ("Refactor this long function into smaller functions", "realworld"),
    ("Refactor to extract reusable helper function", "realworld"),
    ("Refactor this to reduce cyclomatic complexity", "realworld"),
    ("Refactor to remove duplicate code", "realworld"),
    ("Refactor this to follow single responsibility principle", "realworld"),
    ("Refactor to improve function cohesion", "realworld"),
    ("Refactor to reduce function parameters", "realworld"),
    ("Refactor to use parameter object pattern", "realworld"),
    ("Refactor to return early and reduce nesting", "realworld"),
    ("Refactor to use guard clauses", "realworld"),
    # Class refactoring
    ("Refactor this class to follow SRP", "realworld"),
    ("Refactor to extract interface", "realworld"),
    ("Refactor to use composition over inheritance", "realworld"),
    ("Refactor to apply dependency injection", "realworld"),
    ("Refactor to extract abstract base class", "realworld"),
    ("Refactor to use mixin classes", "realworld"),
    ("Refactor to implement protocol/interface", "realworld"),
    ("Refactor to use dataclass", "realworld"),
    ("Refactor to use named tuples", "realworld"),
    ("Refactor to use Pydantic models", "realworld"),
    # Pattern application
    ("Refactor to use factory pattern", "realworld"),
    ("Refactor to use builder pattern", "realworld"),
    ("Refactor to use strategy pattern", "realworld"),
    ("Refactor to use observer pattern", "realworld"),
    ("Refactor to use decorator pattern", "realworld"),
    ("Refactor to use adapter pattern", "realworld"),
    ("Refactor to use facade pattern", "realworld"),
    ("Refactor to use command pattern", "realworld"),
    ("Refactor to use template method pattern", "realworld"),
    ("Refactor to use state pattern", "realworld"),
    # Error handling refactoring
    ("Refactor to use proper exception handling", "realworld"),
    ("Refactor to use custom exceptions", "realworld"),
    ("Refactor to use exception chaining", "realworld"),
    ("Refactor to use context managers", "realworld"),
    ("Refactor to use try-finally properly", "realworld"),
    ("Refactor to handle errors gracefully", "realworld"),
    ("Refactor to add proper error messages", "realworld"),
    ("Refactor to use Result type pattern", "realworld"),
    ("Refactor to use Optional properly", "realworld"),
    ("Refactor to handle None values safely", "realworld"),
    # Async refactoring
    ("Refactor this sync code to async", "realworld"),
    ("Refactor to use asyncio.gather", "realworld"),
    ("Refactor to use async context managers", "realworld"),
    ("Refactor to use async generators", "realworld"),
    ("Refactor to handle async exceptions", "realworld"),
    ("Refactor to use async semaphore", "realworld"),
    ("Refactor to use async queue", "realworld"),
    ("Refactor to batch async operations", "realworld"),
    ("Refactor to use connection pooling", "realworld"),
    ("Refactor to add proper timeouts", "realworld"),
    # Performance refactoring
    ("Refactor to improve time complexity", "realworld"),
    ("Refactor to reduce memory usage", "realworld"),
    ("Refactor to use lazy evaluation", "realworld"),
    ("Refactor to add caching", "realworld"),
    ("Refactor to use memoization", "realworld"),
    ("Refactor to batch database queries", "realworld"),
    ("Refactor to use bulk operations", "realworld"),
    ("Refactor to stream large data", "realworld"),
    ("Refactor to use generators", "realworld"),
    ("Refactor to reduce allocations", "realworld"),
    # Code organization
    ("Refactor to organize by feature", "realworld"),
    ("Refactor to separate concerns", "realworld"),
    ("Refactor to create proper modules", "realworld"),
    ("Refactor to improve imports", "realworld"),
    ("Refactor to use __all__ properly", "realworld"),
    ("Refactor to add type hints", "realworld"),
    ("Refactor to improve docstrings", "realworld"),
    ("Refactor to follow PEP 8 style", "realworld"),
    ("Refactor to use constants", "realworld"),
    ("Refactor to use enums", "realworld"),
    # Testing refactoring
    ("Refactor to make code testable", "realworld"),
    ("Refactor to inject dependencies", "realworld"),
    ("Refactor to use test doubles", "realworld"),
    ("Refactor to improve test readability", "realworld"),
    ("Refactor to reduce test duplication", "realworld"),
    ("Refactor to use test fixtures", "realworld"),
    ("Refactor to parameterize tests", "realworld"),
    ("Refactor to improve test isolation", "realworld"),
    ("Refactor to add proper assertions", "realworld"),
    ("Refactor to improve test naming", "realworld"),
    # API refactoring
    ("Refactor API to be RESTful", "realworld"),
    ("Refactor to use proper HTTP methods", "realworld"),
    ("Refactor to add proper status codes", "realworld"),
    ("Refactor to improve error responses", "realworld"),
    ("Refactor to add pagination", "realworld"),
    ("Refactor to add filtering", "realworld"),
    ("Refactor to add sorting", "realworld"),
    ("Refactor to version the API", "realworld"),
    ("Refactor to add rate limiting", "realworld"),
    ("Refactor to improve API documentation", "realworld"),
]

# =============================================================================
# TESTING PROMPTS (~80 entries)
# =============================================================================

TESTING_PROMPTS: List[Tuple[str, str]] = [
    # Unit tests
    ("Write unit tests for this function", "realworld"),
    ("Write tests that cover edge cases", "realworld"),
    ("Write tests for error handling", "realworld"),
    ("Write tests for boundary conditions", "realworld"),
    ("Write tests with parametrized inputs", "realworld"),
    ("Write tests for async function", "realworld"),
    ("Write tests using mocks", "realworld"),
    ("Write tests using fixtures", "realworld"),
    ("Write tests for class methods", "realworld"),
    ("Write tests for private methods", "realworld"),
    # Test patterns
    ("Write tests following AAA pattern", "realworld"),
    ("Write tests following Given-When-Then", "realworld"),
    ("Write tests using test doubles", "realworld"),
    ("Write tests using dependency injection", "realworld"),
    ("Write tests with factory functions", "realworld"),
    ("Write tests with builder pattern", "realworld"),
    ("Write tests using object mother", "realworld"),
    ("Write tests with test data builder", "realworld"),
    ("Write tests using snapshot testing", "realworld"),
    ("Write tests using golden files", "realworld"),
    # Integration tests
    ("Write integration tests for database", "realworld"),
    ("Write integration tests for API", "realworld"),
    ("Write integration tests for service", "realworld"),
    ("Write integration tests for queue", "realworld"),
    ("Write integration tests for cache", "realworld"),
    ("Write integration tests for file system", "realworld"),
    ("Write integration tests for external API", "realworld"),
    ("Write integration tests for authentication", "realworld"),
    ("Write integration tests for authorization", "realworld"),
    ("Write integration tests for messaging", "realworld"),
    # E2E tests
    ("Write E2E test for user flow", "realworld"),
    ("Write E2E test for checkout process", "realworld"),
    ("Write E2E test for registration", "realworld"),
    ("Write E2E test for login", "realworld"),
    ("Write E2E test for search", "realworld"),
    ("Write E2E test for navigation", "realworld"),
    ("Write E2E test for form submission", "realworld"),
    ("Write E2E test for file upload", "realworld"),
    ("Write E2E test for notifications", "realworld"),
    ("Write E2E test for error handling", "realworld"),
    # Mock/Stub creation
    ("Write a mock for database connection", "realworld"),
    ("Write a mock for HTTP client", "realworld"),
    ("Write a mock for file system", "realworld"),
    ("Write a mock for cache", "realworld"),
    ("Write a mock for message queue", "realworld"),
    ("Write a stub for external service", "realworld"),
    ("Write a fake for in-memory storage", "realworld"),
    ("Write a spy for method calls", "realworld"),
    ("Write a mock for async operation", "realworld"),
    ("Write a mock for streaming data", "realworld"),
    # Fixtures
    ("Write pytest fixture for database", "realworld"),
    ("Write pytest fixture for test client", "realworld"),
    ("Write pytest fixture for auth token", "realworld"),
    ("Write pytest fixture for test data", "realworld"),
    ("Write pytest fixture for temp files", "realworld"),
    ("Write pytest fixture with cleanup", "realworld"),
    ("Write pytest fixture with scope", "realworld"),
    ("Write pytest fixture with parameters", "realworld"),
    ("Write pytest fixture for dependency injection", "realworld"),
    ("Write pytest fixture for configuration", "realworld"),
    # Performance tests
    ("Write performance test for API", "realworld"),
    ("Write load test for endpoint", "realworld"),
    ("Write stress test for service", "realworld"),
    ("Write benchmark for function", "realworld"),
    ("Write performance test for database query", "realworld"),
    ("Write memory usage test", "realworld"),
    ("Write CPU profiling test", "realworld"),
    ("Write latency measurement test", "realworld"),
    ("Write throughput test", "realworld"),
    ("Write scalability test", "realworld"),
    # Property-based tests
    ("Write property-based test for function", "realworld"),
    ("Write hypothesis test for invariants", "realworld"),
    ("Write fuzzing test for parser", "realworld"),
    ("Write random input test", "realworld"),
    ("Write boundary value test", "realworld"),
    ("Write equivalence class test", "realworld"),
    ("Write mutation test", "realworld"),
    ("Write contract test", "realworld"),
    ("Write pact test for API", "realworld"),
    ("Write schema validation test", "realworld"),
]

# =============================================================================
# ALGORITHM IMPLEMENTATION PROMPTS (~70 entries)
# =============================================================================

ALGORITHM_IMPLEMENTATION_PROMPTS: List[Tuple[str, str]] = [
    # Sorting algorithms
    ("Implement quicksort algorithm", "algorithm"),
    ("Implement mergesort algorithm", "algorithm"),
    ("Implement heapsort algorithm", "algorithm"),
    ("Implement insertion sort", "algorithm"),
    ("Implement selection sort", "algorithm"),
    ("Implement bubble sort", "algorithm"),
    ("Implement radix sort", "algorithm"),
    ("Implement counting sort", "algorithm"),
    ("Implement bucket sort", "algorithm"),
    ("Implement timsort algorithm", "algorithm"),
    # Search algorithms
    ("Implement binary search", "algorithm"),
    ("Implement linear search", "algorithm"),
    ("Implement jump search", "algorithm"),
    ("Implement interpolation search", "algorithm"),
    ("Implement exponential search", "algorithm"),
    ("Implement ternary search", "algorithm"),
    ("Implement fibonacci search", "algorithm"),
    ("Implement hash-based search", "algorithm"),
    ("Implement A* search algorithm", "algorithm"),
    ("Implement beam search", "algorithm"),
    # Graph algorithms
    ("Implement DFS for graph", "algorithm"),
    ("Implement BFS for graph", "algorithm"),
    ("Implement Dijkstra's algorithm", "algorithm"),
    ("Implement Bellman-Ford algorithm", "algorithm"),
    ("Implement Floyd-Warshall algorithm", "algorithm"),
    ("Implement Kruskal's algorithm", "algorithm"),
    ("Implement Prim's algorithm", "algorithm"),
    ("Implement topological sort", "algorithm"),
    ("Implement strongly connected components", "algorithm"),
    ("Implement cycle detection in graph", "algorithm"),
    # Dynamic programming
    ("Implement longest common subsequence", "algorithm"),
    ("Implement longest increasing subsequence", "algorithm"),
    ("Implement edit distance", "algorithm"),
    ("Implement knapsack problem", "algorithm"),
    ("Implement coin change problem", "algorithm"),
    ("Implement matrix chain multiplication", "algorithm"),
    ("Implement rod cutting problem", "algorithm"),
    ("Implement subset sum problem", "algorithm"),
    ("Implement palindrome partitioning", "algorithm"),
    ("Implement word break problem", "algorithm"),
    # Tree algorithms
    ("Implement binary tree traversal", "algorithm"),
    ("Implement BST insertion and deletion", "algorithm"),
    ("Implement AVL tree rotation", "algorithm"),
    ("Implement red-black tree", "algorithm"),
    ("Implement segment tree", "algorithm"),
    ("Implement Fenwick tree", "algorithm"),
    ("Implement trie data structure", "algorithm"),
    ("Implement suffix tree", "algorithm"),
    ("Implement lowest common ancestor", "algorithm"),
    ("Implement tree diameter", "algorithm"),
    # String algorithms
    ("Implement KMP string matching", "algorithm"),
    ("Implement Rabin-Karp algorithm", "algorithm"),
    ("Implement Z algorithm", "algorithm"),
    ("Implement suffix array", "algorithm"),
    ("Implement Aho-Corasick algorithm", "algorithm"),
    ("Implement Manacher's algorithm", "algorithm"),
    ("Implement longest repeated substring", "algorithm"),
    ("Implement string hashing", "algorithm"),
    ("Implement rolling hash", "algorithm"),
    ("Implement minimum window substring", "algorithm"),
    # Other algorithms
    ("Implement union-find data structure", "algorithm"),
    ("Implement LRU cache", "algorithm"),
    ("Implement LFU cache", "algorithm"),
    ("Implement bloom filter", "algorithm"),
    ("Implement consistent hashing", "algorithm"),
    ("Implement skip list", "algorithm"),
    ("Implement reservoir sampling", "algorithm"),
    ("Implement Fisher-Yates shuffle", "algorithm"),
    ("Implement median of stream", "algorithm"),
    ("Implement sliding window maximum", "algorithm"),
]

# =============================================================================
# API INTEGRATION PROMPTS (~60 entries)
# =============================================================================

API_INTEGRATION_PROMPTS: List[Tuple[str, str]] = [
    # HTTP clients
    ("Write HTTP GET request with requests", "realworld"),
    ("Write HTTP POST with JSON body", "realworld"),
    ("Write async HTTP request with aiohttp", "realworld"),
    ("Write HTTP request with retries", "realworld"),
    ("Write HTTP request with timeout", "realworld"),
    ("Write HTTP request with headers", "realworld"),
    ("Write HTTP request with authentication", "realworld"),
    ("Write HTTP request with query params", "realworld"),
    ("Write multipart form upload", "realworld"),
    ("Write streaming download", "realworld"),
    # Database operations
    ("Write PostgreSQL connection and query", "realworld"),
    ("Write MySQL database operations", "realworld"),
    ("Write MongoDB CRUD operations", "realworld"),
    ("Write Redis cache operations", "realworld"),
    ("Write Elasticsearch queries", "realworld"),
    ("Write SQLite database operations", "realworld"),
    ("Write database transaction", "realworld"),
    ("Write connection pool setup", "realworld"),
    ("Write database migration", "realworld"),
    ("Write bulk insert operation", "realworld"),
    # Message queues
    ("Write RabbitMQ producer and consumer", "realworld"),
    ("Write Kafka producer and consumer", "realworld"),
    ("Write Redis pub/sub", "realworld"),
    ("Write SQS message handling", "realworld"),
    ("Write Celery task", "realworld"),
    ("Write async queue consumer", "realworld"),
    ("Write dead letter queue handler", "realworld"),
    ("Write message retry logic", "realworld"),
    ("Write message deduplication", "realworld"),
    ("Write message batching", "realworld"),
    # Authentication
    ("Write OAuth2 authentication", "realworld"),
    ("Write JWT token handling", "realworld"),
    ("Write API key authentication", "realworld"),
    ("Write Basic authentication", "realworld"),
    ("Write SAML authentication", "realworld"),
    ("Write OpenID Connect flow", "realworld"),
    ("Write token refresh logic", "realworld"),
    ("Write session management", "realworld"),
    ("Write CORS configuration", "realworld"),
    ("Write CSRF protection", "realworld"),
    # Cloud services
    ("Write S3 file upload", "realworld"),
    ("Write AWS Lambda function", "realworld"),
    ("Write DynamoDB operations", "realworld"),
    ("Write Google Cloud Storage upload", "realworld"),
    ("Write Azure Blob storage", "realworld"),
    ("Write cloud function trigger", "realworld"),
    ("Write SNS notification", "realworld"),
    ("Write CloudWatch logging", "realworld"),
    ("Write Secrets Manager access", "realworld"),
    ("Write Parameter Store access", "realworld"),
    # Third-party APIs
    ("Write Stripe payment integration", "realworld"),
    ("Write SendGrid email sending", "realworld"),
    ("Write Twilio SMS sending", "realworld"),
    ("Write GitHub API integration", "realworld"),
    ("Write Slack webhook", "realworld"),
    ("Write Discord bot command", "realworld"),
    ("Write OpenAI API call", "realworld"),
    ("Write Google Maps API", "realworld"),
    ("Write Twitter API integration", "realworld"),
    ("Write Shopify API integration", "realworld"),
]

# =============================================================================
# DATA STRUCTURE PROMPTS (~50 entries)
# =============================================================================

DATA_STRUCTURE_PROMPTS: List[Tuple[str, str]] = [
    # List operations
    ("Write function to reverse a list in place", "realworld"),
    ("Write function to rotate list by k positions", "realworld"),
    ("Write function to partition list by pivot", "realworld"),
    ("Write function to merge two sorted lists", "realworld"),
    ("Write function to find intersection of lists", "realworld"),
    ("Write function to find union of lists", "realworld"),
    ("Write function to remove duplicates from list", "realworld"),
    ("Write function to find kth largest element", "realworld"),
    ("Write function to find majority element", "realworld"),
    ("Write function to move zeros to end", "realworld"),
    # Dictionary operations
    ("Write function to merge dictionaries", "realworld"),
    ("Write function to invert dictionary", "realworld"),
    ("Write function to flatten nested dict", "realworld"),
    ("Write function to deep copy dictionary", "realworld"),
    ("Write function to filter dictionary by keys", "realworld"),
    ("Write function to group by key", "realworld"),
    ("Write function to count occurrences", "realworld"),
    ("Write function to sort dict by value", "realworld"),
    ("Write function to find common keys", "realworld"),
    ("Write function to diff two dicts", "realworld"),
    # Set operations
    ("Write function to find symmetric difference", "realworld"),
    ("Write function to check subset", "realworld"),
    ("Write function to find power set", "realworld"),
    ("Write function to check disjoint sets", "realworld"),
    ("Write function to find missing elements", "realworld"),
    ("Write function to find unique elements", "realworld"),
    ("Write function to cartesian product", "realworld"),
    ("Write function to find permutations", "realworld"),
    ("Write function to find combinations", "realworld"),
    ("Write function to generate subsets", "realworld"),
    # Tree operations
    ("Write function to traverse binary tree", "realworld"),
    ("Write function to find tree height", "realworld"),
    ("Write function to check balanced tree", "realworld"),
    ("Write function to find LCA in tree", "realworld"),
    ("Write function to serialize tree", "realworld"),
    ("Write function to deserialize tree", "realworld"),
    ("Write function to invert binary tree", "realworld"),
    ("Write function to validate BST", "realworld"),
    ("Write function to find path sum", "realworld"),
    ("Write function to flatten tree to list", "realworld"),
    # Graph operations
    ("Write function to detect cycle in graph", "realworld"),
    ("Write function to find connected components", "realworld"),
    ("Write function to check bipartite graph", "realworld"),
    ("Write function to find shortest path", "realworld"),
    ("Write function to clone graph", "realworld"),
    ("Write function to topological sort", "realworld"),
    ("Write function to find bridges in graph", "realworld"),
    ("Write function to find articulation points", "realworld"),
    ("Write function to check if graph is tree", "realworld"),
    ("Write function to find minimum spanning tree", "realworld"),
]

# =============================================================================
# FILE I/O PROMPTS (~30 entries)
# =============================================================================

FILE_IO_PROMPTS: List[Tuple[str, str]] = [
    # Reading files
    ("Write function to read CSV file to dict", "realworld"),
    ("Write function to parse JSON file", "realworld"),
    ("Write function to read YAML config", "realworld"),
    ("Write function to read XML file", "realworld"),
    ("Write function to read log file", "realworld"),
    ("Write function to read binary file", "realworld"),
    ("Write function to read Excel file", "realworld"),
    ("Write function to read Parquet file", "realworld"),
    ("Write function to stream large file", "realworld"),
    ("Write function to read file in chunks", "realworld"),
    # Writing files
    ("Write function to write CSV file", "realworld"),
    ("Write function to write JSON with indentation", "realworld"),
    ("Write function to write YAML config", "realworld"),
    ("Write function to write XML file", "realworld"),
    ("Write function to append to log file", "realworld"),
    ("Write function to write binary file", "realworld"),
    ("Write function to write Excel file", "realworld"),
    ("Write function to write Parquet file", "realworld"),
    ("Write function to stream write large file", "realworld"),
    ("Write function to atomic file write", "realworld"),
    # File operations
    ("Write function to copy file", "realworld"),
    ("Write function to move file", "realworld"),
    ("Write function to delete file safely", "realworld"),
    ("Write function to compress file", "realworld"),
    ("Write function to decompress archive", "realworld"),
    ("Write function to calculate file hash", "realworld"),
    ("Write function to watch file changes", "realworld"),
    ("Write function to lock file for writing", "realworld"),
    ("Write function to create temp file", "realworld"),
    ("Write function to walk directory tree", "realworld"),
]

# =============================================================================
# STRING MANIPULATION PROMPTS (~20 entries)
# =============================================================================

STRING_MANIPULATION_PROMPTS: List[Tuple[str, str]] = [
    ("Write function to validate email format", "realworld"),
    ("Write function to extract URLs from text", "realworld"),
    ("Write function to parse phone numbers", "realworld"),
    ("Write function to format currency", "realworld"),
    ("Write function to slugify string", "realworld"),
    ("Write function to truncate with ellipsis", "realworld"),
    ("Write function to camelCase to snake_case", "realworld"),
    ("Write function to title case string", "realworld"),
    ("Write function to strip HTML tags", "realworld"),
    ("Write function to escape special chars", "realworld"),
    ("Write function to find all occurrences", "realworld"),
    ("Write function to replace multiple patterns", "realworld"),
    ("Write function to split by multiple delimiters", "realworld"),
    ("Write function to join with last separator", "realworld"),
    ("Write function to pad string to width", "realworld"),
    ("Write function to word wrap text", "realworld"),
    ("Write function to highlight matches", "realworld"),
    ("Write function to redact sensitive data", "realworld"),
    ("Write function to template string", "realworld"),
    ("Write function to pluralize word", "realworld"),
]

# =============================================================================
# MATHEMATICAL PROMPTS (~10 entries)
# =============================================================================

MATHEMATICAL_PROMPTS: List[Tuple[str, str]] = [
    ("Write function to calculate mean absolute deviation", "realworld"),
    ("Write function to find prime factorization", "realworld"),
    ("Write function to calculate GCD and LCM", "realworld"),
    ("Write function to check if number is prime", "realworld"),
    ("Write function to generate prime numbers", "realworld"),
    ("Write function to calculate power mod", "realworld"),
    ("Write function to solve quadratic equation", "realworld"),
    ("Write function to calculate combinations", "realworld"),
    ("Write function to calculate statistics", "realworld"),
    ("Write function to convert number bases", "realworld"),
]

# =============================================================================
# GENERAL CODING PROMPTS (~10 entries)
# =============================================================================

GENERAL_CODING_PROMPTS: List[Tuple[str, str]] = [
    ("Write a CLI tool with argparse", "realworld"),
    ("Write a config file parser", "realworld"),
    ("Write a logging setup function", "realworld"),
    ("Write a plugin system", "realworld"),
    ("Write a state machine", "realworld"),
    ("Write a decorator with arguments", "realworld"),
    ("Write a context manager", "realworld"),
    ("Write a thread-safe singleton", "realworld"),
    ("Write an event emitter", "realworld"),
    ("Write a task scheduler", "realworld"),
]

# =============================================================================
# COMPLETE CORPUS BUILDER
# =============================================================================


def build_complete_corpus() -> list:
    """Build the complete corpus with all categories.

    Returns:
        List of tuples (prompt, category, source)
    """
    from victor.agent.prompt_corpus_registry import PromptCategory

    corpus = []

    # Function completion (25%)
    for prompt, source in FUNCTION_COMPLETION_PROMPTS:
        corpus.append((prompt, PromptCategory.FUNCTION_COMPLETION, source))

    # Code debugging (20%)
    for prompt, source in CODE_DEBUGGING_PROMPTS:
        corpus.append((prompt, PromptCategory.CODE_DEBUGGING, source))

    # Code explanation (12%)
    for prompt, source in CODE_EXPLANATION_PROMPTS:
        corpus.append((prompt, PromptCategory.CODE_EXPLANATION, source))

    # Code refactoring (10%)
    for prompt, source in CODE_REFACTORING_PROMPTS:
        corpus.append((prompt, PromptCategory.CODE_REFACTORING, source))

    # Testing (8%)
    for prompt, source in TESTING_PROMPTS:
        corpus.append((prompt, PromptCategory.TESTING, source))

    # Algorithm implementation (7%)
    for prompt, source in ALGORITHM_IMPLEMENTATION_PROMPTS:
        corpus.append((prompt, PromptCategory.ALGORITHM_IMPLEMENTATION, source))

    # API integration (6%)
    for prompt, source in API_INTEGRATION_PROMPTS:
        corpus.append((prompt, PromptCategory.API_INTEGRATION, source))

    # Data structure (5%)
    for prompt, source in DATA_STRUCTURE_PROMPTS:
        corpus.append((prompt, PromptCategory.DATA_STRUCTURE, source))

    # File I/O (3%)
    for prompt, source in FILE_IO_PROMPTS:
        corpus.append((prompt, PromptCategory.FILE_IO, source))

    # String manipulation (2%)
    for prompt, source in STRING_MANIPULATION_PROMPTS:
        corpus.append((prompt, PromptCategory.STRING_MANIPULATION, source))

    # Mathematical (1%)
    for prompt, source in MATHEMATICAL_PROMPTS:
        corpus.append((prompt, PromptCategory.MATHEMATICAL, source))

    # General coding (1%)
    for prompt, source in GENERAL_CODING_PROMPTS:
        corpus.append((prompt, PromptCategory.GENERAL_CODING, source))

    return corpus


def get_corpus_stats() -> dict:
    """Get statistics about the corpus.

    Returns:
        Dictionary with category counts and percentages
    """
    total = (
        len(FUNCTION_COMPLETION_PROMPTS)
        + len(CODE_DEBUGGING_PROMPTS)
        + len(CODE_EXPLANATION_PROMPTS)
        + len(CODE_REFACTORING_PROMPTS)
        + len(TESTING_PROMPTS)
        + len(ALGORITHM_IMPLEMENTATION_PROMPTS)
        + len(API_INTEGRATION_PROMPTS)
        + len(DATA_STRUCTURE_PROMPTS)
        + len(FILE_IO_PROMPTS)
        + len(STRING_MANIPULATION_PROMPTS)
        + len(MATHEMATICAL_PROMPTS)
        + len(GENERAL_CODING_PROMPTS)
    )

    return {
        "total_entries": total,
        "categories": {
            "function_completion": {
                "count": len(FUNCTION_COMPLETION_PROMPTS),
                "percentage": len(FUNCTION_COMPLETION_PROMPTS) / total * 100,
            },
            "code_debugging": {
                "count": len(CODE_DEBUGGING_PROMPTS),
                "percentage": len(CODE_DEBUGGING_PROMPTS) / total * 100,
            },
            "code_explanation": {
                "count": len(CODE_EXPLANATION_PROMPTS),
                "percentage": len(CODE_EXPLANATION_PROMPTS) / total * 100,
            },
            "code_refactoring": {
                "count": len(CODE_REFACTORING_PROMPTS),
                "percentage": len(CODE_REFACTORING_PROMPTS) / total * 100,
            },
            "testing": {
                "count": len(TESTING_PROMPTS),
                "percentage": len(TESTING_PROMPTS) / total * 100,
            },
            "algorithm_implementation": {
                "count": len(ALGORITHM_IMPLEMENTATION_PROMPTS),
                "percentage": len(ALGORITHM_IMPLEMENTATION_PROMPTS) / total * 100,
            },
            "api_integration": {
                "count": len(API_INTEGRATION_PROMPTS),
                "percentage": len(API_INTEGRATION_PROMPTS) / total * 100,
            },
            "data_structure": {
                "count": len(DATA_STRUCTURE_PROMPTS),
                "percentage": len(DATA_STRUCTURE_PROMPTS) / total * 100,
            },
            "file_io": {
                "count": len(FILE_IO_PROMPTS),
                "percentage": len(FILE_IO_PROMPTS) / total * 100,
            },
            "string_manipulation": {
                "count": len(STRING_MANIPULATION_PROMPTS),
                "percentage": len(STRING_MANIPULATION_PROMPTS) / total * 100,
            },
            "mathematical": {
                "count": len(MATHEMATICAL_PROMPTS),
                "percentage": len(MATHEMATICAL_PROMPTS) / total * 100,
            },
            "general_coding": {
                "count": len(GENERAL_CODING_PROMPTS),
                "percentage": len(GENERAL_CODING_PROMPTS) / total * 100,
            },
        },
    }


# =============================================================================
# DYNAMIC BENCHMARK LOADER
# =============================================================================


def load_humaneval_prompts() -> List[Tuple[str, str, str]]:
    """Load prompts from HumanEval benchmark via HuggingFace.

    HumanEval contains 164 programming problems with docstrings.

    Returns:
        List of (prompt, category, source) tuples

    Raises:
        ImportError: If datasets library is not installed
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompts = []

        for item in dataset:
            # HumanEval prompts are function signatures with docstrings
            prompt = item["prompt"]
            # Classify based on docstring content
            category = _classify_humaneval_prompt(prompt)
            prompts.append((prompt, category, f"humaneval:{item['task_id']}"))

        return prompts

    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")


def load_mbpp_prompts() -> List[Tuple[str, str, str]]:
    """Load prompts from MBPP benchmark via HuggingFace.

    MBPP (Mostly Basic Python Problems) contains ~1000 programming problems.

    Returns:
        List of (prompt, category, source) tuples

    Raises:
        ImportError: If datasets library is not installed
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("google-research-datasets/mbpp", split="test")
        prompts = []

        for item in dataset:
            # MBPP prompts are natural language descriptions
            prompt = item["text"]
            category = _classify_mbpp_prompt(prompt)
            prompts.append((prompt, category, f"mbpp:{item['task_id']}"))

        return prompts

    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")


def _classify_humaneval_prompt(prompt: str) -> TaskCategory:
    """Classify a HumanEval prompt into a category.

    Args:
        prompt: The function signature and docstring

    Returns:
        TaskCategory enum value
    """
    prompt_lower = prompt.lower()

    # Classify based on keywords in the prompt
    if any(kw in prompt_lower for kw in ["sort", "sorted", "order", "arrange"]):
        return TaskCategory.ALGORITHM_IMPLEMENTATION
    elif any(kw in prompt_lower for kw in ["list", "array", "tuple", "dict", "stack", "queue"]):
        return TaskCategory.DATA_STRUCTURE
    elif any(kw in prompt_lower for kw in ["string", "char", "substr", "concat", "split"]):
        return TaskCategory.STRING_MANIPULATION
    elif any(kw in prompt_lower for kw in ["math", "sum", "product", "prime", "factor", "gcd"]):
        return TaskCategory.MATHEMATICAL
    elif any(kw in prompt_lower for kw in ["file", "read", "write", "path"]):
        return TaskCategory.FILE_IO
    else:
        return TaskCategory.FUNCTION_COMPLETION


def _classify_mbpp_prompt(prompt: str) -> TaskCategory:
    """Classify an MBPP prompt into a category.

    Args:
        prompt: The natural language problem description

    Returns:
        TaskCategory enum value
    """
    prompt_lower = prompt.lower()

    # Classify based on keywords - order matters (more specific first)
    # File I/O - use specific patterns to avoid false positives from "write a function"
    if any(
        kw in prompt_lower
        for kw in ["read file", "write file", "open file", "file path", "directory"]
    ):
        return TaskCategory.FILE_IO
    # Algorithm patterns
    elif any(
        kw in prompt_lower
        for kw in ["sort", "search", "binary", "merge", "quick", "heap", "permut"]
    ):
        return TaskCategory.ALGORITHM_IMPLEMENTATION
    # Mathematical patterns
    elif any(
        kw in prompt_lower
        for kw in [
            "calculate",
            "sum of",
            "product of",
            "factorial",
            "prime",
            "fibonacci",
            "gcd",
            "lcm",
            "power",
            "square",
        ]
    ):
        return TaskCategory.MATHEMATICAL
    # String patterns
    elif any(
        kw in prompt_lower
        for kw in [
            "string",
            "character",
            "substring",
            "regex",
            "palindrome",
            "anagram",
            "vowel",
            "consonant",
        ]
    ):
        return TaskCategory.STRING_MANIPULATION
    # Data structure patterns
    elif any(
        kw in prompt_lower
        for kw in [
            "list",
            "array",
            "dictionary",
            "tuple",
            "set",
            "stack",
            "queue",
            "matrix",
            "element",
        ]
    ):
        return TaskCategory.DATA_STRUCTURE
    # Testing patterns
    elif any(kw in prompt_lower for kw in ["test", "assert", "unit test"]):
        return TaskCategory.TESTING
    else:
        return TaskCategory.FUNCTION_COMPLETION


def build_extended_corpus_with_benchmarks(
    include_humaneval: bool = True,
    include_mbpp: bool = True,
) -> List[Tuple[str, str, str]]:
    """Build an extended corpus that includes benchmark datasets.

    This function loads the static corpus and optionally adds prompts
    from HumanEval and MBPP benchmarks when the datasets library is available.

    Args:
        include_humaneval: Include HumanEval benchmark prompts
        include_mbpp: Include MBPP benchmark prompts

    Returns:
        List of (prompt, category, source) tuples
    """
    # Start with static corpus
    corpus = build_complete_corpus()

    # Try to add HumanEval
    if include_humaneval:
        try:
            humaneval_prompts = load_humaneval_prompts()
            corpus.extend(humaneval_prompts)
        except ImportError:
            pass  # datasets not available, skip
        except Exception:
            pass  # Other error, skip silently

    # Try to add MBPP
    if include_mbpp:
        try:
            mbpp_prompts = load_mbpp_prompts()
            corpus.extend(mbpp_prompts)
        except ImportError:
            pass  # datasets not available, skip
        except Exception:
            pass  # Other error, skip silently

    return corpus


def get_benchmark_stats() -> dict:
    """Get statistics about available benchmark datasets.

    Returns:
        Dictionary with benchmark availability and counts
    """
    stats = {
        "static_corpus": get_corpus_stats(),
        "benchmarks": {
            "humaneval": {"available": False, "count": 0},
            "mbpp": {"available": False, "count": 0},
        },
    }

    try:
        humaneval = load_humaneval_prompts()
        stats["benchmarks"]["humaneval"] = {
            "available": True,
            "count": len(humaneval),
        }
    except (ImportError, Exception):
        pass

    try:
        mbpp = load_mbpp_prompts()
        stats["benchmarks"]["mbpp"] = {
            "available": True,
            "count": len(mbpp),
        }
    except (ImportError, Exception):
        pass

    return stats
