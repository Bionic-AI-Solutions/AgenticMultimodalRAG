#!/usr/bin/env python3
"""
Simple test script to verify FastAPI MCP integration.
Run this after starting the FastAPI application.
"""
import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
MCP_BASE_URL = f"{BASE_URL}/mcp"


def test_mcp_tools_list() -> bool:
    """Test listing available MCP tools"""
    print("\n=== Testing MCP Tools List ===")
    try:
        # Test our wrapper endpoint
        response = requests.get(f"{BASE_URL}/agent/tools/list", timeout=5)
        print(f"GET /agent/tools/list: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Found {data.get('count', 0)} tools")
            if data.get("tools"):
                print(f"  Sample tools: {[t.get('name', 'unknown') for t in data['tools'][:5]]}")
            return True
        else:
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def test_mcp_direct_endpoints() -> bool:
    """Test direct MCP protocol endpoints"""
    print("\n=== Testing Direct MCP Endpoints ===")
    success = True
    
    # Try different possible endpoint paths
    endpoints_to_try = [
        f"{MCP_BASE_URL}/tools/list",
        f"{MCP_BASE_URL}/tools",
        f"{BASE_URL}/mcp/tools/list",
    ]
    
    for endpoint in endpoints_to_try:
        try:
            response = requests.get(endpoint, timeout=5)
            print(f"GET {endpoint}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Success! Response: {json.dumps(data, indent=2)[:200]}...")
                return True
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print("  None of the MCP endpoints responded successfully")
    return False


def test_fastapi_mcp_tool_call() -> bool:
    """Test calling a tool via FastAPI MCP"""
    print("\n=== Testing FastAPI MCP Tool Call ===")
    
    # Try calling health_check tool
    tool_call_endpoints = [
        f"{MCP_BASE_URL}/tools/call",
        f"{MCP_BASE_URL}/call",
        f"{BASE_URL}/mcp/tools/call",
    ]
    
    payload = {
        "name": "health_check",
        "arguments": {}
    }
    
    for endpoint in tool_call_endpoints:
        try:
            response = requests.post(endpoint, json=payload, timeout=5)
            print(f"POST {endpoint}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Success! Response: {json.dumps(data, indent=2)[:200]}...")
                return True
            else:
                print(f"  Error: {response.text[:200]}")
        except Exception as e:
            print(f"  Exception: {e}")
            continue
    
    print("  Tool call failed on all endpoints")
    return False


def test_agent_executor_with_mcp_tool() -> bool:
    """Test agent executor using FastAPI MCP tool"""
    print("\n=== Testing Agent Executor with MCP Tool ===")
    
    plan = {
        "plan": [
            {
                "step_id": 1,
                "type": "tool_call",
                "modality": "text",
                "parameters": {
                    "tool": "fastapi_mcp",
                    "tool_name": "health_check",
                    "arguments": {}
                },
                "dependencies": [],
                "trace": {}
            }
        ],
        "traceability": True,
        "app_id": "testapp",
        "user_id": "testuser"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/agent/execute",
            json=plan,
            timeout=10
        )
        print(f"POST /agent/execute: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Execution completed")
            print(f"  Trace length: {len(data.get('trace', []))}")
            if data.get("final_result"):
                result_preview = json.dumps(data["final_result"], indent=2)[:300]
                print(f"  Final result preview: {result_preview}...")
            return True
        else:
            print(f"  Error: {response.text[:300]}")
            return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def main():
    """Run all tests"""
    print("FastAPI MCP Integration Test")
    print("=" * 50)
    print(f"Base URL: {BASE_URL}")
    print(f"MCP Base URL: {MCP_BASE_URL}")
    
    # Check if server is running
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"\nERROR: Server health check failed ({health_response.status_code})")
            print("Make sure the FastAPI application is running!")
            sys.exit(1)
        print("\n✓ Server is running")
    except Exception as e:
        print(f"\nERROR: Cannot connect to server: {e}")
        print("Make sure the FastAPI application is running at http://localhost:8000")
        sys.exit(1)
    
    results = []
    
    # Run tests
    results.append(("MCP Tools List", test_mcp_tools_list()))
    results.append(("Direct MCP Endpoints", test_mcp_direct_endpoints()))
    results.append(("MCP Tool Call", test_fastapi_mcp_tool_call()))
    results.append(("Agent Executor with MCP", test_agent_executor_with_mcp_tool()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
        print("\nNote: If FastAPI MCP is not installed, some tests may fail.")
        print("Install it with: poetry add fastapi-mcp")
        sys.exit(1)


if __name__ == "__main__":
    main()

