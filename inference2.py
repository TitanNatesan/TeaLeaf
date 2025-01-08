import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ==============================
# 1. Configuration and Setup
# ==============================

# Define class mappings based on your dataset
class_to_idx = {
    'background': 0,
    'rsc': 1,
    'looper': 2,
    'rsm': 3,
    'thrips': 4,
    'jassid': 5,
    'tmb': 6,
    'healthy': 7
}
num_classes = len(class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Define colors for visualization (extend as needed)
class_colors = {
    1: (255, 0, 0),    # Red
    2: (0, 255, 0),    # Green
    3: (0, 0, 255),    # Blue
    4: (255, 255, 0),  # Yellow
    5: (255, 0, 255),  # Magenta
    6: (0, 255, 255),  # Cyan
    7: (128, 0, 128)   # Purple
}

# Paths (Update these paths accordingly)
model_weights_path = r'D:\Titan\Projects\MobileAppONNX\maskrcnn_finetuned.pth'
image_path = r'D:\Titan\Projects\MobileAppONNX\data\GOPR3178.JPG'
output_combined_path = r'D:\Titan\Projects\MobileAppONNX\output\combined_segmentation.jpg'

# Confidence threshold for displaying predictions
CONFIDENCE_THRESHOLD = 0.3  # Lowered to ensure more predictions are shown

# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# ==============================
# 2. Model Initialization and Loading
# ==============================

def get_model_instance_segmentation(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=num_classes
    )
    return model

model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from {model_weights_path}")

# ==============================
# 3. Image Preprocessing
# ==============================

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    return image, input_tensor

# ==============================
# 4. Inference Function
# ==============================

def run_inference(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs

# ==============================
# 5. Visualization Function (Combined Instance + Semantic Segmentation)
# ==============================

def save_combined_segmentation(image, outputs, output_path):
    """
    Save a combined instance and semantic segmentation image with class labels.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 78)
    except IOError:
        print("Arial font not found. Using default font.")
        font = ImageFont.load_default()

    pred_scores = outputs[0]['scores'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_masks = outputs[0]['masks'].cpu().numpy()  # [N, 1, H, W]

    print(f"Total Predictions: {len(pred_scores)}")

    # Create an empty overlay for masks with transparency
    mask_overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))

    for i, (score, label, box, mask) in enumerate(zip(pred_scores, pred_labels, pred_boxes, pred_masks)):
        if score < CONFIDENCE_THRESHOLD:
            continue
        
        class_name = idx_to_class[label]
        box_color = class_colors[label]
        print(f"Prediction {i + 1}: {class_name} with score {score:.2f}")

        # Generate mask
        mask = mask[0] > 0.5
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
        
        # Create a colored mask with transparency
        colored_mask = Image.new('RGBA', image.size, box_color + (128,))
        mask_overlay = Image.alpha_composite(mask_overlay, Image.composite(colored_mask, mask_overlay, mask_image))

        # Draw bounding box
        draw.rectangle(box.tolist(), outline=box_color, width=4)

        # Draw label and score
        text = f"{class_name}: {score:.2f}"
        text_width, text_height = font.getbbox(text)[2:]  # Use font.getbbox() to get width and height

        # Draw a rectangle behind the text for better visibility
        draw.rectangle(
            [(box[0], box[1] - text_height - 5), (box[0] + text_width, box[1])],
            fill=box_color
        )

        # Draw the text
        draw.text((box[0], box[1] - text_height - 5), text, fill='black', font=font)

    # Composite the mask overlay onto the original image
    combined_image = Image.alpha_composite(image.convert('RGBA'), mask_overlay)
    combined_image.convert('RGB').save(output_path)
    print(f"Combined segmentation image saved at {output_path}")


def save_highest_confidence_segmentation(image, outputs, output_path):
    """
    Save an image with only the highest confidence prediction visualized.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 78)
    except IOError:
        print("Arial font not found. Using default font.")
        font = ImageFont.load_default()

    pred_scores = outputs[0]['scores'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_masks = outputs[0]['masks'].cpu().numpy()  # [N, 1, H, W]

    print(f"Total Predictions: {len(pred_scores)}")

    if len(pred_scores) == 0:
        print("No predictions to display.")
        return

    # Get the index of the highest confidence prediction
    max_conf_index = np.argmax(pred_scores)
    max_score = pred_scores[max_conf_index]
    if max_score < CONFIDENCE_THRESHOLD:
        print(f"No prediction above confidence threshold ({CONFIDENCE_THRESHOLD}).")
        return

    label = pred_labels[max_conf_index]
    box = pred_boxes[max_conf_index]
    mask = pred_masks[max_conf_index][0] > 0.5
    class_name = idx_to_class[label]
    box_color = class_colors.get(label, (255, 0, 0))  # Default to red if no color is specified

    print(f"Highest Confidence Prediction: {class_name} with score {max_score:.2f}")

    # Resize mask to match the image size
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)

    # Create a colored mask with transparency
    mask_overlay = Image.new('RGBA', image.size, box_color + (128,))
    mask_overlay = Image.alpha_composite(
        Image.new('RGBA', image.size, (255, 255, 255, 0)),
        Image.composite(mask_overlay, Image.new('RGBA', image.size), mask_image)
    )

    # Draw bounding box
    draw.rectangle(box.tolist(), outline=box_color, width=4)

    # Draw label and score
    text = f"{class_name}: {max_score:.2f}"
    text_width, text_height = font.getbbox(text)[2:]  # Use font.getbbox() to get width and height

    # Draw a rectangle behind the text for better visibility
    draw.rectangle(
        [(box[0], box[1] - text_height - 5), (box[0] + text_width, box[1])],
        fill=box_color
    )

    # Draw the text
    draw.text((box[0], box[1] - text_height - 5), text, fill='black', font=font)

    # Composite the mask overlay onto the original image
    combined_image = Image.alpha_composite(image.convert('RGBA'), mask_overlay)
    combined_image.convert('RGB').save(output_path)
    print(f"Segmented image with highest confidence saved at {output_path}")


# ==============================
# 6. Main Execution Flow
# ==============================

def main():
    original_image, input_tensor = preprocess_image(image_path)
    outputs = run_inference(model, input_tensor)
    save_highest_confidence_segmentation(original_image.copy(), outputs, output_combined_path)

if __name__ == "__main__":
    main()
