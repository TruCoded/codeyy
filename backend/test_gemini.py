import os
from dotenv import load_dotenv
import asyncio
from ai_service import detect_language

async def fast_test():
    load_dotenv()
    print("Running FAST connection test...")
    try:
        # Just test a simple language detection
        lang = await detect_language("print('test')")
        print(f"PASS! Detected: {lang}")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    asyncio.run(fast_test())
