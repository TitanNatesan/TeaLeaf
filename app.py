import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import base64

# ==============================
# 1. Configuration and Setup
# ==============================

# Define class mappings, colors, and remedies
class_to_idx = {'background': 0, 'rsc': 1, 'looper': 2, 'rsm': 3, 'thrips': 4, 'jassid': 5, 'tmb': 6, 'healthy': 7}
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_colors = {
    1: (255, 0, 0),    # Red
    2: (0, 255, 0),    # Green
    3: (0, 0, 255),    # Blue
    4: (255, 255, 0),  # Yellow
    5: (255, 0, 255),  # Magenta
    6: (0, 255, 255),  # Cyan
    7: (128, 0, 128)   # Purple
}

remedies = {
    "rsc": "Apply neem oil or insecticidal soap to control this pest.",
    "looper": "Use Bacillus thuringiensis (Bt) as a natural pesticide.",
    "rsm": "Maintain high humidity and use acaricides to manage mites.",
    "thrips": "Introduce beneficial insects like lacewings or use blue sticky traps.",
    "jassid": "Spray with imidacloprid or other systemic insecticides.",
    "tmb": "Remove affected leaves and apply copper fungicides.",
    "healthy": "No action needed; the plant is healthy!"
} 

# Pest recommendations
pest_recommendations = {
    "tmb": {
        "name": "Tea Mosquito Bug (TMB)",
        "symptoms": [
            "Brown or black feeding spots on shoots.",
            "Wilted or damaged shoots due to nymph feeding.",
            "Presence of eggs and early nymph stages inserted in shoots."
        ],
        "control_methods": {
            "biological": [
                "Encourage natural predators like Sycanus collaris (Reduviid bug), Chrysoperla carnea, Mallada boninensis (Lacewings), Oxyopes spp. (Spiders), and praying mantis."
            ],
            "chemical": [
                "Use systemic insecticides alternately: Thiamethoxam 25 WG, Clothianidin 50 WDG, Thiacloprid 21.7% SC, and Neem extract (Azadirachtin 5% W/W)."
            ],
            "mechanical": [
                "Regularly remove infested shoots during plucking.",
                "Maintain a close plucking schedule to remove eggs and nymphs.",
                "Remove alternate host plants near plantations.",
                "Prune and skiff bushes during cold weather."
            ]
        }
    },
    "rsm": {
        "name": "Red Spider Mite (RSM)",
        "symptoms": [
            "Presence of red spots and webbing on the underside of leaves.",
            "Leaves turn bronze or rusty red and fall off.",
            "Infestation is more severe in unshaded, waterlogged areas."
        ],
        "control_methods": {
            "biological": [
                "Conserve predatory insects and mites like Phytoseiid mites (Amblyseius sp., Cunaxa sp.), Ladybird beetles (Stethorus sp., Scymnus sp.), and Lacewings (Mallada sp., Chrysopa sp.)."
            ],
            "chemical": [
                "Use acaricides alternately: Propargite 57 EC, Fenazaquin 10 EC, Spiromesifen 240 SC, and Hexythiazox 5.45 EC."
            ],
            "mechanical": [
                "Maintain shade trees at recommended spacing to reduce mite buildup.",
                "Remove alternate host plants near plantations.",
                "Improve drainage to prevent waterlogging."
            ]
        }
    },
    "rsc": {
        "name": "Red Slug Caterpillar (RSC)",
        "symptoms": [
            "Feeding damage on young leaves.",
            "Defoliation of bushes during severe infestations.",
            "Presence of caterpillars with a distinctive red body and slug-like appearance."
        ],
        "control_methods": {
            "biological": [
                "Encourage natural enemies like birds, parasitic wasps, and pathogenic fungi and bacteria in the soil."
            ],
            "chemical": [
                "Apply insecticides like Emamectin Benzoate 5% SG and Flubendiamide 20% WG."
            ],
            "mechanical": [
                "Manual collection and destruction of caterpillars.",
                "Prune and clean bushes to remove pupae from crevices."
            ]
        }
    },
    "looper": {
        "name": "Looper Caterpillar",
        "symptoms": [
            "Defoliation of bushes due to feeding by caterpillars.",
            "Caterpillars are visible hanging from leaves using silken threads.",
            "Eggs laid in clusters on cracks of shade tree bark."
        ],
        "control_methods": {
            "biological": [
                "Encourage natural enemies like Cotesia ruficrus (Parasitoid wasp), Sycanus collaris (Predatory bug), spiders (Oxyopes shweta), and entomopathogenic nematodes (Steinernema sp., Heterorhabditis sp.)."
            ],
            "chemical": [
                "Apply insecticides alternately: Emamectin Benzoate 5% SG, Quinalphos 25 EC, and Deltamethrin 10 EC."
            ],
            "mechanical": [
                "Manual removal of caterpillars, moths, and chrysalids.",
                "Light scrapping of shade tree bark to destroy eggs.",
                "Use light traps during the evening to attract and kill moths."
            ]
        }
    },
    "thrips": {
        "name": "Thrips",
        "symptoms": [
            "Leaves show silvering and curling due to feeding.",
            "Leaf tips may turn yellowish or brown.",
            "Both adult and larval thrips can be found on leaves."
        ],
        "control_methods": {
            "biological": [
                "Encourage natural predators like predatory thrips (Aeolothrips intermedius, Mymarothrips garuda), spiders, and dragonflies."
            ],
            "chemical": [
                "Use systemic insecticides alternately: Thiamethoxam 25 WG, Clothianidin 50 WDG, and Bifenthrin 8 SC."
            ],
            "mechanical": [
                "Use yellow sticky traps (45 cm wide) to attract and trap thrips.",
                "Maintain a shade level of 60% in the plantation."
            ]
        }
    },
    "jassid": {
        "name": "Jassid",
        "symptoms": [
            "Yellowing and curling of leaf edges.",
            "Leaves show brown spots and withering in severe infestations.",
            "Both adults and nymphs feed on the underside of leaves."
        ],
        "control_methods": {
            "biological": [
                "Conserve natural predators like ladybeetles (Stethorus sp., Scymnus sp.), predatory bugs (Anthocoris sp., Orius sp.), and lacewings (Chrysopa sp., Chrysoperla sp.)."
            ],
            "chemical": [
                "Use systemic insecticides alternately: Thiamethoxam 25 WG, Clothianidin 50 WDG, and Spirotetramat 15.31% OD."
            ],
            "mechanical": [
                "Use light traps and yellow sticky traps to monitor and control populations.",
                "Caustic wash the trunk and stir soil around the collar region to kill pupae."
            ]
        }
    }
}


# Model path and device configuration
model_weights_path = r'D:\Titan\Projects\MobileAppONNX\maskrcnn_finetuned.pth'  # Replace with your model path
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CONFIDENCE_THRESHOLD = 0.35

# Initialize FastAPI app
app = FastAPI()

origins = ["*"]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allowed origins
    allow_credentials=True,          # Allow cookies and credentials
    allow_methods=["*"],             # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],             # Allow all HTTP headers
)

# ==============================
# 2. Model Initialization
# ==============================

def get_model_instance_segmentation(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=in_features_mask, dim_reduced=hidden_layer, num_classes=num_classes
    )
    return model

model = get_model_instance_segmentation(num_classes=len(class_to_idx))
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from {model_weights_path}")

# ==============================
# 3. Helper Functions
# ==============================

def preprocess_image(image_data: bytes):
    """Load image from bytes and preprocess."""
    image = Image.open(BytesIO(image_data)).convert("RGB")
    input_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    return image, input_tensor

def run_inference(model, input_tensor):
    """Run inference on the input tensor."""
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs

def save_highest_confidence_segmentation(image, outputs):
    """Generate a segmentation image for the highest confidence prediction."""
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

    if len(pred_scores) == 0:
        print("No predictions available.")
        return None, None

    # Find the prediction with the highest confidence score
    max_conf_index = np.argmax(pred_scores)
    max_score = pred_scores[max_conf_index]

    # Only process if the highest score exceeds the threshold
    if max_score < CONFIDENCE_THRESHOLD:
        print(f"No predictions above confidence threshold ({CONFIDENCE_THRESHOLD}).")
        return None, None

    label = pred_labels[max_conf_index]
    box = pred_boxes[max_conf_index]
    mask = pred_masks[max_conf_index][0] > 0.5
    class_name = idx_to_class[label]
    box_color = class_colors.get(label, (255, 0, 0))  # Default to red if no color specified

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
    text_width, text_height = font.getbbox(text)[2:]
    draw.rectangle([(box[0], box[1] - text_height - 5), (box[0] + text_width, box[1])], fill=box_color)
    draw.text((box[0], box[1] - text_height - 5), text, fill='black', font=font)

    # Composite the mask overlay onto the original image
    combined_image = Image.alpha_composite(image.convert('RGBA'), mask_overlay)
    combined_image = combined_image.convert('RGB')

    # Save the result to a buffer
    output_buffer = BytesIO()
    combined_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)
    return output_buffer, class_name

# ==============================
# 4. FastAPI Endpoints
# ==============================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to handle image upload and return prediction plus processed image."""
    try:
        image_data = await file.read()
        original_image, input_tensor = preprocess_image(image_data)
        outputs = run_inference(model, input_tensor)
        output_buffer, class_name = save_highest_confidence_segmentation(original_image.copy(), outputs)

        if output_buffer is None:
            return JSONResponse(content={"message": "No predictions above confidence threshold."}, status_code=206)

        # Convert processed image to base64
        processed_img_b64 = base64.b64encode(output_buffer.read()).decode("utf-8")

        recommendation = pest_recommendations.get(class_name, {
            "name": "Unknown Pest",
            "symptoms": ["No data available."],
            "control_methods": {"biological": [], "chemical": [], "mechanical": []}
        })

        response = {
            "prediction": class_name,
            "remedy": remedies.get(class_name, "No remedy available."),
            "recommendation": recommendation,
            "processed_image": processed_img_b64
        }

        return JSONResponse(content=response, headers={"X-Prediction": class_name})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
