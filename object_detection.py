


from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Define the path to the local model directory
model_dir = "/app/models/detr-finetuned-food-packaging"

# Load the model and processor from the local directory
processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModelForObjectDetection.from_pretrained(model_dir)

# Load the local image
image_path = "/app/example.png"
image = Image.open(image_path).convert("RGB")
image = np.array(image)

# Process the image
inputs = processor(images=[image], return_tensors="pt")

# Perform object detection
outputs = model(**inputs)

# Convert outputs to Pascal VOC format (xmin, ymin, xmax, ymax)
target_sizes = torch.tensor([image.shape[:2]])
results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

# Plot the image and the bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(image)

# Add bounding boxes to the image
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    xmin, ymin, xmax, ymax = box
    width, height = xmax - xmin, ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(xmin, ymin, f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", color='white', fontsize=12, backgroundcolor='red')

# Ensure the output directory exists
output_dir = "/app/output"
os.makedirs(output_dir, exist_ok=True)

# Save the result
output_path = os.path.join(output_dir, "output.png")
plt.savefig(output_path)
print(f"Saved the image with bounding boxes to {output_path}")
