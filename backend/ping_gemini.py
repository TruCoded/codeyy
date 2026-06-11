import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.5-flash")

try:
    print("Connecting to Gemini...")
    response = model.generate_content("Say 'Gemini is ready!'")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
