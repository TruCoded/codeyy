import os
from dotenv import load_dotenv
import asyncio
from ai_service import detect_language

async def test_new_sdk():
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    print(f"Testing with key prefix: {key[:5]}...")
    
    try:
        print(f"Testing with Gemini Service...")
        lang = await detect_language("print('hello')")
        print(f"Success! Detected: {lang}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_new_sdk())
