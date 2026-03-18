#!/usr/bin/env python3
"""
Lab assistant agent — answers questions using an LLM with tools.

Usage:
    uv run agent.py "What does REST stand for?"

Output:
    {
      "answer": "...",
      "source": "wiki/rest-api.md#what-is-rest",
      "tool_calls": [...]
    }
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# Load LLM configuration from .env.agent.secret
load_dotenv(".env.agent.secret")

# LLM configuration
LLM_API_BASE = os.getenv("LLM_API_BASE")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

# Backend API configuration (load from .env.docker.secret)
load_dotenv(".env.docker.secret", override=True)
LMS_API_KEY = os.getenv("LMS_API_KEY")
AGENT_API_BASE_URL = os.getenv("AGENT_API_BASE_URL", "http://localhost:42002")

# Project root for tool path resolution
PROJECT_ROOT = Path(__file__).parent.resolve()

# Maximum tool calls per query
MAX_TOOL_CALLS = 15

# Tool definitions for LLM function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file from the project repository. Use this to read file contents after discovering relevant files with list_files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root (e.g., 'wiki/rest-api.md')",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a given path in the project repository. Use this to discover files in a directory. ALWAYS use this before read_file to find the correct file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path from project root (e.g., 'wiki/', 'backend/', 'backend/app/routers/')",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_api",
            "description": "Query the backend API to get real-time data or perform actions. Use this for questions about database contents, statistics, or system state. Do NOT use for static documentation questions — use read_file or list_files for those.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method (GET, POST, PUT, DELETE, etc.)",
                    },
                    "path": {
                        "type": "string",
                        "description": "API endpoint path (e.g., '/items/', '/analytics/completion-rate')",
                    },
                    "body": {
                        "type": "string",
                        "description": "JSON request body for POST/PUT requests (optional)",
                    },
                    "auth": {
                        "type": "boolean",
                        "description": "Whether to include Authorization header (default: true). Set to false to test unauthenticated access.",
                    },
                },
                "required": ["method", "path"],
            },
        },
    },
]

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the project repository and backend API.

You have access to these tools:
- list_files(path): List files/directories at a given path
- read_file(path): Read contents of a file
- query_api(method, path, body, auth): Query the backend API for real-time data

CRITICAL RULES:
1. NEVER guess file paths. ALWAYS use list_files first to discover the correct file path before calling read_file.
2. When asked about backend code, ALWAYS start by listing backend/app/ or backend/app/routers/ to find the actual file names.
3. Read files COMPLETELY - do not skip any content. The answer may be at the beginning, middle, or end.
4. For wiki questions, list wiki/ first, then read the relevant file COMPLETELY.

Decision workflow:
1. For static documentation questions (e.g., "What is REST?", "How to protect a branch?"):
   - FIRST: list_files("wiki/") to find relevant files
   - THEN: read_file the relevant file(s) COMPLETELY
   - Answer based on the full file content

2. For data-dependent questions (e.g., "How many items?", "How many learners?"):
   - Use query_api to fetch the data
   - If the response contains a list, COUNT the items and return the number
   - Always provide the actual count from the API response

3. For system facts (e.g., "What framework?", "What port?"):
   - FIRST: list_files("backend/") to find the main files
   - THEN: read_file on source code (backend/app/main.py, docker-compose.yml)

4. To test unauthenticated access (e.g., "What status code without auth?"):
   - Use query_api with auth=false
   - Report the exact status code returned

5. For bug diagnosis questions (e.g., "What error?", "Which operation is risky?"):
   - FIRST: Query the API to reproduce the error
   - THEN: ALWAYS read backend/app/routers/analytics.py (or the relevant source file)
   - Look for dangerous operations:
     * Division: x / y where y could be zero (e.g., len(x) / len(y) when both empty)
     * Sorting None: sorted(None, ...) causes TypeError
     * Accessing attributes on None: x.attribute when x could be None
   - Explain the exact line and why it fails
   - You MUST use BOTH query_api AND read_file for these questions

6. For comparison questions (e.g., "Compare error handling in ETL vs API"):
   - FIRST: Read the ETL code (backend/app/etl.py or similar)
   - THEN: Read the API router code (backend/app/routers/*.py)
   - Compare how each handles errors:
     * Does it use try/except?
     * Does it log errors or silently ignore?
     * Does it raise exceptions or return error values?
   - Describe both approaches and highlight differences

Rules:
- Always provide the source file path where you found the answer (for wiki/code questions)
- For API queries, include the endpoint path in your answer
- At the end of your answer, add a line: "Source: <file-path>" (e.g., "Source: backend/app/routers/analytics.py")
- For bug diagnosis, always cite the source file where the bug is located
- For comparison questions, cite ALL files you read (e.g., "Source: backend/app/etl.py, backend/app/routers/analytics.py")
- If you can't find the answer after exploring, say so honestly
- Don't make up information not present in the files or API responses
- When you find the answer, respond with the answer and source, do not make additional tool calls
"""

def is_safe_path(requested_path: str) -> bool:
    """Check if the requested path is within the project directory."""
    if os.path.isabs(requested_path):
        return False
    if ".." in requested_path:
        return False
    full_path = (PROJECT_ROOT / requested_path).resolve()
    return str(full_path).startswith(str(PROJECT_ROOT))


def read_file(path: str) -> str:
    """Read contents of a file from the project repository.
    
    FIX #1: No truncation - reads the entire file since Qwen has large context.
    """
    if not is_safe_path(path):
        return f"Error: Invalid path '{path}'. Path traversal not allowed."
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return f"Error: File '{path}' does not exist."
    if not file_path.is_file():
        return f"Error: '{path}' is not a file."
    try:
        # Read the ENTIRE file - no truncation
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


def list_files(path: str) -> str:
    """List files and directories at a given path."""
    if not is_safe_path(path):
        return f"Error: Invalid path '{path}'. Path traversal not allowed."
    dir_path = PROJECT_ROOT / path
    if not dir_path.exists():
        return f"Error: Directory '{path}' does not exist."
    if not dir_path.is_dir():
        return f"Error: '{path}' is not a directory."
    try:
        entries = sorted(dir_path.iterdir())
        return "\n".join([entry.name for entry in entries])
    except Exception as e:
        return f"Error listing directory: {e}"


def query_api(method: str, path: str, body: str | None = None, auth: bool = True) -> str:
    """Query the backend API with optional authentication.
    
    FIX #3: Properly handles auth parameter - when auth=False, no Authorization
    header is sent, allowing testing of 401/403 responses.
    """
    import json as json_module
    
    base_url = AGENT_API_BASE_URL.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    url = f"{base_url}{path}"

    headers = {"Content-Type": "application/json"}
    
    # FIX #3: Only add Authorization header if auth=True AND LMS_API_KEY is set
    if auth:
        if not LMS_API_KEY:
            return json_module.dumps({
                "status_code": 500,
                "body": "Error: LMS_API_KEY not configured. Check .env.docker.secret."
            })
        headers["Authorization"] = f"Bearer {LMS_API_KEY}"
    # When auth=False, we intentionally do NOT add the Authorization header
    # This allows testing unauthenticated access

    try:
        with httpx.Client() as client:
            json_body = None
            if body:
                json_body = json_module.loads(body)

            method = method.upper()
            if method == "GET":
                response = client.get(url, headers=headers, timeout=30.0)
            elif method == "POST":
                response = client.post(url, headers=headers, json=json_body, timeout=30.0)
            elif method == "PUT":
                response = client.put(url, headers=headers, json=json_body, timeout=30.0)
            elif method == "DELETE":
                response = client.delete(url, headers=headers, timeout=30.0)
            else:
                return json_module.dumps({
                    "status_code": 400,
                    "body": f"Error: Unsupported HTTP method '{method}'"
                })

            try:
                response_body = response.json()
            except:
                response_body = response.text

            return json_module.dumps({
                "status_code": response.status_code,
                "body": response_body
            })

    except httpx.TimeoutException:
        return json_module.dumps({"status_code": 504, "body": "Request timeout"})
    except httpx.ConnectError:
        return json_module.dumps({"status_code": 503, "body": f"Could not connect to {AGENT_API_BASE_URL}"})
    except Exception as e:
        return json_module.dumps({"status_code": 500, "body": f"Error: {str(e)}"})


# Map of tool names to functions
TOOLS = {
    "read_file": read_file,
    "list_files": list_files,
    "query_api": query_api,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name with the given arguments."""
    if name not in TOOLS:
        return f"Error: Unknown tool '{name}'"
    func = TOOLS[name]
    try:
        return func(**arguments)
    except Exception as e:
        return f"Error executing {name}: {e}"


def call_llm(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Send messages to the LLM and return the response."""
    if not LLM_API_BASE or not LLM_API_KEY:
        raise RuntimeError("LLM not configured. Check .env.agent.secret")

    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL or "qwen3-coder-plus",
        "messages": messages,
        "temperature": 0.7,
    }

    if tools:
        payload["tools"] = tools

    for attempt in range(3):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0)

    raise RuntimeError("Failed to get LLM response after max retries")


def extract_json_from_response(text: str) -> dict[str, Any] | None:
    """FIX #2: Robustly extract JSON from LLM response.
    
    Handles:
    - Raw JSON: {"answer": "...", "source": "..."}
    - Markdown wrapped: ```json {...} ```
    - Conversational text before/after JSON
    - Multiple JSON objects (takes the first valid one)
    """
    text = text.strip()
    
    # Try parsing as raw JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data:
            return data
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON within markdown code blocks
    # Pattern 1: ```json {...} ```
    json_block_pattern = r'```(?:json)?\s*({\s*"answer"\s*:.*?})\s*```'
    matches = re.findall(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "answer" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    # Pattern 2: Any JSON object with "answer" key
    # This is more greedy but handles cases where LLM adds text
    json_object_pattern = r'({\s*"answer"\s*:[^{}]*(?:{[^{}]*}[^{}]*)*})'
    matches = re.findall(json_object_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "answer" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    # Pattern 3: Try to find any balanced JSON object
    # Use a stack-based approach to find balanced braces
    start_idx = text.find('{')
    while start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            candidate = text[start_idx:end_idx]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "answer" in data:
                    return data
            except json.JSONDecodeError:
                pass
        
        start_idx = text.find('{', start_idx + 1)
    
    return None


def extract_source_from_answer(answer: str) -> str:
    """Extract source file path from the answer text."""
    source = ""
    
    if "source:" in answer.lower():
        matches = re.findall(
            r"source:\s*`?([a-zA-Z0-9_/.-]+\.(py|md|json|yml|yaml))",
            answer,
            re.IGNORECASE,
        )
        if matches:
            file_paths = [m[0] for m in matches]
            # Prefer .py files for code questions
            for path in file_paths:
                if path.endswith(".py"):
                    source = path
                    break
            if not source:
                source = file_paths[0]
    
    return source


def run_agentic_loop(question: str) -> dict[str, Any]:
    """Main agentic loop: processes question with tool calling."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": question})
    tool_calls_log = []

    for _ in range(MAX_TOOL_CALLS):
        try:
            response = call_llm(messages, tools=TOOL_DEFINITIONS)
        except Exception as e:
            return {
                "answer": f"Error calling LLM: {e}",
                "source": "",
                "tool_calls": tool_calls_log,
            }

        message = response["choices"][0]["message"]

        if "tool_calls" in message and message["tool_calls"]:
            messages.append({"role": "assistant", "tool_calls": message["tool_calls"]})

            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                result = execute_tool(tool_name, tool_args)

                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })
        else:
            # LLM returned a final answer (no tool calls)
            answer_text = message.get("content", "")
            
            # FIX #2: Try to extract structured JSON from the response
            extracted = extract_json_from_response(answer_text)
            
            if extracted:
                answer = extracted.get("answer", "")
                source = extracted.get("source", "")
            else:
                # Fallback: parse answer text and extract source
                answer = answer_text
                source = extract_source_from_answer(answer_text)
            
            # Ensure we have at least an answer
            if not answer:
                answer = answer_text
            
            return {
                "answer": answer,
                "source": source,
                "tool_calls": tool_calls_log,
            }

    return {
        "answer": "Unable to find answer within maximum tool calls.",
        "source": "",
        "tool_calls": tool_calls_log,
    }


def main():
    if len(sys.argv) != 2:
        print('Usage: uv run agent.py "<question>"', file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1]
    result = run_agentic_loop(question)

    # Only JSON to stdout, nothing else
    print(json.dumps(result))


if __name__ == "__main__":
    main()
