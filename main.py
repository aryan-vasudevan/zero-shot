import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import supervision as sv

load_dotenv()

# Load Gemini 2.5 and prompt
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

MODEL_NAME = "gemini-2.5-flash-preview-05-20"
TEMPERATURE = 0.5

IMAGE_PATH = "sample.jpg"

# Object detection prompt
# PROMPT = "Detect the helmets." + \
# "Output a JSON list of bounding boxes where each entry contains the 2D bounding box in the key \"box_2d\", " + \
# "and the text label in the key \"label\". Use descriptive labels."

# Instance segmentation prompt
PROMPT = "Give the segmentation masks for the motorcycles." + \
"Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", " + \
"the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels."

# Image and response
image = Image.open(IMAGE_PATH)
width, height = image.size
target_height = int(1024 * height / width)
resized_image = image.resize((1024, target_height), Image.Resampling.LANCZOS)

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=[PROMPT, resized_image],
    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        safety_settings=safety_settings,
        thinking_config=types.ThinkingConfig(
          thinking_budget=0
        )
    )
)

# Overlay image
resolution_wh = image.size

detections = sv.Detections.from_vlm(
    vlm=sv.VLM.GOOGLE_GEMINI_2_5,
    result=response.text,
    resolution_wh=resolution_wh
)

thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh) / 3

box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    smart_position=True,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_position=sv.Position.CENTER
)

# Object detection annotations
# annotated = image
# for annotator in (box_annotator, label_annotator):
#     annotated = annotator.annotate(scene=annotated, detections=detections)

# sv.plot_image(annotated)

# Instance segemntation annotations
masks_annotator = sv.MaskAnnotator()

annotated = image
for annotator in (box_annotator, label_annotator, masks_annotator):
    annotated = annotator.annotate(scene=annotated, detections=detections)

sv.plot_image(annotated)