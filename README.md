 * # Project Documentation
 *
 * This project leverages a Mask R-CNN model (with ResNet50-FPN backbones) for object detection
 * and segmentation, specifically targeting pest detection in tea plants. The application uses FastAPI 
 * to provide a RESTful endpoint for uploading images and returning pest detection results, including 
 * recommendations and remedies.
 *
 * ## Project Information
 * - **Model**: Mask R-CNN with pretrained backbone, tailored to detect various tea plant pests.
 * - **Core Logic**:
 *   - A FastAPI endpoint receives image uploads.
 *   - Images are transformed into tensors for inference.
 *   - Model outputs (boxes, labels, and masks) are filtered by confidence threshold.
 *   - The highest-confidence pest is returned with recommended remedies and pest details.
 *
 * ## Setup and Dependencies
 * - **Python**: Requires Python 3.x.
 * - **Dependencies**: Install packages such as FastAPI, PyTorch, TorchVision, Pillow, and NumPy.
 *   This can be done using pip or a similar package manager.
 * - **Model Weights**: Ensure the trained model weights file is placed in the expected path.
 *
 * ## Running the Service
 * 1. Install the required dependencies.
 * 2. Launch the FastAPI application, for example:
 *
 *     uvicorn app:app --host 0.0.0.0 --port 8000
 *
 * 3. Use an HTTP POST request to the `/predict/` endpoint with an image to obtain detection results.
 *
 * ## Extending the Project
 * - Add or modify classes: Expand the list of detectable pests by retraining or fine-tuning the model.
 * - Tweak confidence threshold: Adjust this value to control detection sensitivity.
 * - Integrate with a frontend: Implement any web or mobile client to upload images and display results.
 */

## Overview
This project implements a Mask R-CNN model using PyTorch for image segmentation and pest detection in tea plants. A FastAPI application is provided to accept image uploads, perform inference using a trained model, and respond with useful information on the predicted pest class and its recommended remedies.

## Key Features
1. **Model Loading:** The code initializes a Mask R-CNN model, equipped with ResNet50-FPN backbones pretrained on default weights. The model is modified to detect custom classes representing various pests (plus a "healthy" class).
2. **Inference Flow:** Once the app starts, an uploaded image is preprocessed into a tensor, placed on the appropriate device (CPU or GPU), and passed to the initialized Mask R-CNN model. The model returns boxes, labels, masks, and scores.
3. **Confidence Filtering:** Predictions are filtered by a configurable confidence threshold. Only the highest-confidence detection above this threshold is visualized and returned to the client.
4. **Visualization:** Predicted bounding boxes and masks are drawn onto the original image in memory. Different classes are color-coded, and the highest-confidence prediction is highlighted. The processed image is not directly returned but is used to generate supplementary data for the client.
5. **Pest Information:** The application includes a dictionary of remedies and a larger “pest_recommendations” structure describing each pest’s name, symptoms, and control methods. These recommendations are served along with the prediction results.
6. **Endpoints:**
    - `/predict/`: Expects an uploaded image file. Proceeds with inference, applies color mask overlays, and ultimately responds with JSON data describing the pest detection result. The user can use this endpoint to detect pests in uploaded tea plant images.
7. **FastAPI Setup:**
    - A CORS middleware is configured to accept requests from any origin. This allows hosting the service for broad accessibility.
    - The main application code is encapsulated under FastAPI’s recommended structure.

## Implementation Details
1. **Dependencies and Imports:** 
    - PyTorch and TorchVision for the Mask R-CNN model.
    - FastAPI for creating and handling the web service.
    - Pillow and NumPy for image manipulation, bounding box drawing, and mask encoding.
2. **Custom Classes and Utilities:**
    - `get_model_instance_segmentation(num_classes)` constructs the model with a replaced predictor head, suitable for detecting a custom set of classes.
    - `preprocess_image(image_data)` handles the byte data from the uploaded image and transforms it into a tensor.
    - `run_inference(model, input_tensor)` simply runs the model’s forward pass in evaluation mode with no gradient tracking.
    - `save_highest_confidence_segmentation(image, outputs)` draws masks and bounding boxes on the single most confident prediction above the threshold. It returns an in-memory buffer and the predicted class name if a valid detection is found.
3. **JSON Response Structure:**
    - `prediction`: The detected class name for the highest-confidence prediction.
    - `remedy`: A short textual remedy for the detected pest.
    - `recommendation`: A nested dictionary containing more detailed information such as symptoms, biological, chemical, and mechanical control methods.

## Usage
1. **Installation:**
    - Install Python and ensure the required packages are present (FastAPI, PyTorch, TorchVision, etc.).
2. **Running the Service:**
    - Place the model weights file in the specified path.
    - Run this FastAPI application (for example, “uvicorn app:app --host 0.0.0.0 --port 8000”).
    - Access the `/predict/` endpoint with an HTTP POST request, attaching an image file (“multipart/form-data”).
3. **Inference Workflow:**
    - The request body carries an image.
    - The app preprocesses and infers the image using the loaded model.
    - If a valid detection is found, the API responds with JSON that includes pest information and recommended actions.

## Project Structure (High-Level)
- **main application file**: Contains the FastAPI logic, definitions for the classes, loading and inference logic, and endpoint configuration.
- **model weights file**: The trained model weights expected at a configured path.
- **requirements**: Lists the necessary dependencies (e.g., FastAPI, PyTorch, TorchVision, Pillow).

## Extending the Project
- **Additional Classes**: If there are new pests or classes to detect, expand the class mappings and retrain or fine-tune the model with suitable annotations.
- **Confidence Threshold**: Tweak the `CONFIDENCE_THRESHOLD` variable to balance detection precision and recall.
- **Frontend Integration**: Optionally integrate with a web or mobile frontend that sends images to the `/predict/` endpoint and receives JSON responses to display results.

## Conclusion
This project leverages Mask R-CNN for object detection and segmentation specifically tailored to tea plant pests. The FastAPI service architecture makes it straightforward to deploy the trained model behind a lightweight REST API. By returning pest details and remedies, the application provides immediate guidance on pest control, simplifying a core agricultural challenge.
