from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("deepseek-ai/deepseek-vl")
model = AutoModelForVision2Seq.from_pretrained("deepseek-ai/deepseek-vl")

image = Image.open("test_image.jpg")
prompt = "¿Esta imagen muestra un incendio o fuego?"

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
respuesta = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(respuesta)