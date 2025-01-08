## Project Documentation

> **Leverages a Mask R-CNN model (with ResNet50-FPN backbones) for object detection and segmentation, specifically targeting pest detection in tea plants.**  
> **Implements a FastAPI endpoint** for uploading images and returning pest detection results, including recommendations and remedies.

## Project Information
- **Model**: Mask R-CNN, fine-tuned to detect various tea plant pests.  
- **Core Logic**:  
    1. FastAPI endpoint receives uploaded images.  
    2. Images are converted into tensors for inference.  
    3. Model outputs (boxes, labels, masks) are filtered by confidence threshold.  
    4. The highest-confidence pest is identified and accompanied by recommended remedies.

## Setup and Dependencies
- **Python**: Works with Python 3.x.  
- **Dependencies**: Requires FastAPI, PyTorch, TorchVision, Pillow, NumPy, etc.  
- **Model Weights**: Place the trained weights file in the correct path.

## Running the Service
1. Install required dependencies.  
2. Launch FastAPI (example: `uvicorn app:app --host 0.0.0.0 --port 8000`).  
3. Send an HTTP POST request with an image to `/predict/` to retrieve detection results.

## Extending the Project
- **Add/Modify Classes**: Expand pest detection by retraining or fine-tuning the model.  
- **Confidence Threshold**: Adjust for desired detection sensitivity.  
- **Frontend Integration**: Use a web or mobile client to upload images and show results.
*/

## Overview
A Mask R-CNN model using PyTorch provides object segmentation and pest detection in tea plants. A FastAPI service handles image uploads, performs inference with a trained model, and responds with pest classes plus remedies.

## Key Features
1. **Model Loading**: Mask R-CNN (ResNet50-FPN) pretrained and modified for custom pest classes.  
2. **Inference Flow**: Image → Tensor → Model → Filtered Results (boxes, labels, masks, scores).  
3. **Confidence Filtering**: Focus on highest-confidence predictions above a chosen threshold.  
4. **Visualization**: Overlays for bounding boxes and masks in-memory, highlighting the top detection.  
5. **Pest Information**: Central dictionary of remedies and detailed “pest_recommendations.”  
6. **Endpoints**:  
     - `/predict/`: Upload image, receive JSON with pest detection details.  
7. **FastAPI Setup**: CORS middleware allows broad accessibility, simplifying client integration.

## Implementation Details
1. **Dependencies & Imports**: PyTorch, TorchVision, FastAPI, Pillow, NumPy.  
2. **Custom Classes & Utilities**:  
     - `get_model_instance_segmentation(num_classes)` builds the model with a custom predictor head.  
     - `preprocess_image(image_data)` processes uploaded images into tensors.  
     - `run_inference(model, input_tensor)` runs the model in evaluation mode without gradient tracking.  
     - `save_highest_confidence_segmentation(image, outputs)` annotates only the top prediction above threshold.  
3. **JSON Response Structure**:  
     - `prediction`: Highest-confidence pest class.  
     - `remedy`: Short textual remedy.  
     - `recommendation`: Detailed pest info (symptoms, control methods).

## Usage
1. **Installation**: Python environment with FastAPI, PyTorch, TorchVision, etc.  
2. **Running the Service**:  
     - Ensure model weights are in place.  
     - `uvicorn app:app --host 0.0.0.0 --port 8000` (example).  
     - POST an image to `/predict/` for inference results.  
3. **Inference Workflow**:  
     - The app preprocesses and runs inference on the image.  
     - A valid detection triggers a JSON response with pest details.

## Project Structure (High-Level)
- **Main app file**: FastAPI logic and endpoint definitions.  
- **Model weights**: Trained file in the expected path.  
- **Requirements**: Dependencies listed (FastAPI, PyTorch, TorchVision, Pillow, etc.).

## Extending the Project
- **Additional Classes**: Add pests/classes and retrain or fine-tune.  
- **Confidence Threshold**: Update `CONFIDENCE_THRESHOLD` for performance balance.  
- **Frontend Integration**: Optionally add a web or mobile client to send images and display results.

## Conclusion
Leverages a Mask R-CNN for tea plant pest detection and segmentation. The FastAPI framework provides a lightweight REST API with direct pest details and remedies, supporting efficient agricultural pest management.
