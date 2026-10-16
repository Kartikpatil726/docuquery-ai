import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv('GEMINI_API_KEY')
if not key or key.startswith('paste'):
    raise SystemExit('GEMINI_API_KEY is not set in .env')

client = genai.Client(api_key=key)
model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
reply = client.models.generate_content(model=model, contents='Reply with exactly: KEY WORKS')
print(model, '->', reply.text)
