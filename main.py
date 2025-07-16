import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

MODEL_NAME = "gemini-2.5-flash-preview-05-20"
TEMPERATURE = 0.5

IMAGE_PATH = "/content/dog-3.jpeg"
PROMPT = "Detect dogs tail. " + \
"Output a JSON list of bounding boxes where each entry contains the 2D bounding box in the key \"box_2d\", " + \
"and the text label in the key \"label\". Use descriptive labels."