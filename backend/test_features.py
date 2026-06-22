import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from backend/.env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from ai_service import analyze_code, refactor_code, generate_comments_for_code

TEST_CODE = """def count_pairs(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                count += 1
    return count
"""

async def test_all():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in backend/.env. Cannot run tests.")
        sys.exit(1)
    
    print("Testing analyze_code with language='auto'...")
    try:
        res = await analyze_code(TEST_CODE, "auto")
        print("[OK] analyze_code Success!")
        print(f"  Detected language: '{res.get('detected_language')}'")
        print(f"  Flowchart returned (length={len(res.get('flowchart', ''))}):")
        print(res.get("flowchart")[:100] + "...")
        print(f"  Time complexity: {res.get('time_complexity')}")
    except Exception as e:
        print(f"[FAIL] analyze_code failed: {e}")
        
    print("\nTesting refactor_code to O(N)...")
    try:
        res = await refactor_code(TEST_CODE, "python", "O(N) time")
        print("[OK] refactor_code Success!")
        print(f"  Refactored code snippet:")
        print(res.get("refactored_code")[:150] + "...")
        print(f"  Explanation summary: {res.get('explanation')[:100]}...")
    except Exception as e:
        print(f"[FAIL] refactor_code failed: {e}")

    print("\nTesting generate_comments_for_code...")
    try:
        res = await generate_comments_for_code(TEST_CODE, "python")
        print("[OK] generate_comments_for_code Success!")
        print(f"  Commented code snippet:")
        print(res.get("commented_code")[:150] + "...")
    except Exception as e:
        print(f"[FAIL] generate_comments_for_code failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_all())
