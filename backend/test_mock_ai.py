import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure backend directory is in path
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import ai_service

# Mock responses from Gemini
MOCK_EXPLAIN_RESPONSE = """
DETECTED_LANGUAGE_START
python
DETECTED_LANGUAGE_END

LINE_EXPLANATIONS_START
Line 1: print('hello') | Explanation: Prints hello.
LINE_EXPLANATIONS_END

BUG_DETECTION_START
BUG_NONE
BUG_DETECTION_END

CORRECTED_CODE_START
CLEAN
CORRECTED_CODE_END

TIME_COMPLEXITY_START
O(1) constant time
TIME_COMPLEXITY_END

SPACE_COMPLEXITY_START
O(1) constant space
SPACE_COMPLEXITY_END

SUGGESTIONS_START
1. Good code structure.
SUGGESTIONS_END

DSA_PATTERN_START
Simple Output Pattern
DSA_PATTERN_END

PLATFORM_PROBLEMS_START
LeetCode | Hello World | https://leetcode.com/problems/hello-world
GeeksforGeeks | Print Hello World | https://www.geeksforgeeks.org/print-hello-world
HackerRank | Say Hello World | https://www.hackerrank.com/challenges/say-hello-world
PLATFORM_PROBLEMS_END

OPTIMIZED_CODE_START
print('hello')
OPTIMIZED_CODE_END

OPTIMIZED_TIME_COMPLEXITY_START
O(1) constant time
OPTIMIZED_TIME_COMPLEXITY_END

OPTIMIZED_SPACE_COMPLEXITY_START
O(1) constant space
OPTIMIZED_SPACE_COMPLEXITY_END

PRACTICE_EXERCISES_START
Q1. Print something else.
PRACTICE_EXERCISES_END

INTERVIEW_QUESTIONS_START
Q: Why print?
A: To show output.
INTERVIEW_QUESTIONS_END

ALGORITHM_START
Step 1. Start. Step 2. Print.
ALGORITHM_END

VIVA_QUESTIONS_START
Q: What is print?
A: A built-in function.
VIVA_QUESTIONS_END
"""

MOCK_DRY_RUN_RESPONSE = """
DRY_RUN_START
[
  {
    "step": 1,
    "line": 1,
    "action": "Prints hello",
    "variables": {},
    "ds_type": "none",
    "ds_data": []
  }
]
DRY_RUN_END

FLOWCHART_START
graph TD
  A("Start"):::startEnd --> B["Print hello"]:::default --> C("End"):::startEnd
FLOWCHART_END

RECURSION_TREE_START
RECURSION_NONE
RECURSION_TREE_END
"""

async def run_mock_test():
    print("Running parallel analyze_code unit test with mocked Gemini client...")
    
    # Mock get_client to return a mock client
    mock_client = MagicMock()
    
    # Mock client.models.generate_content
    # The first call will return MOCK_EXPLAIN_RESPONSE
    # The second call will return MOCK_DRY_RUN_RESPONSE
    mock_response_1 = MagicMock()
    mock_response_1.text = MOCK_EXPLAIN_RESPONSE
    
    mock_response_2 = MagicMock()
    mock_response_2.text = MOCK_DRY_RUN_RESPONSE
    
    # Side effect function to return different mock responses
    call_count = 0
    def mock_generate_content(model, contents, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_response_1
        else:
            return mock_response_2

    mock_client.models.generate_content = mock_generate_content
    
    with patch("ai_service.get_client", return_value=mock_client):
        result = await ai_service.analyze_code("print('hello')", "python")
        
        # Verify the outputs are parsed correctly
        assert result["detected_language"] == "python", f"Expected python, got {result['detected_language']}"
        assert result["time_complexity"] == "O(1) constant time", f"Got {result['time_complexity']}"
        assert result["recursion_tree"] == "RECURSION_NONE", f"Got {result['recursion_tree']}"
        assert "Prints hello" in result["dry_run"], f"Got {result['dry_run']}"
        assert "GeeksforGeeks" in result["platform_problems"], f"Got {result['platform_problems']}"
        assert result["optimized_code"] == "print('hello')", f"Got {result['optimized_code']}"
        assert result["optimized_time_complexity"] == "O(1) constant time", f"Got {result['optimized_time_complexity']}"
        assert result["optimized_space_complexity"] == "O(1) constant space", f"Got {result['optimized_space_complexity']}"
        
        print("[SUCCESS] Unit test passed! Parallel prompt queries executed and parsed correctly.")

if __name__ == "__main__":
    asyncio.run(run_mock_test())
