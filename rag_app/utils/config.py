import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env file if it exists
load_dotenv()

# Retrieve API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "❌ GOOGLE_API_KEY is not set. Please create a .env file or set it in your environment."
    )

# Configure the Gemini model globally
genai.configure(api_key=GOOGLE_API_KEY)
