import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image # Library to load images
import os

# --- Configuration ---
num_classes = 4 # Must match the number of classes you trained on
img_size = 224  # Must match the image size used during training

# IMPORTANT: Update this path to where you saved the downloaded .pth file
model_weights_path = '../resnet_brain_tumor_weights.pth'

# Define the device (use GPU if available locally, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define the Model Architecture ---
# Load the ResNet50 architecture *without* pre-trained weights from the internet
# Because we will load our own trained weights
model = models.resnet50(weights=None)

# Get the number of input features for the classifier layer
num_ftrs = model.fc.in_features

# Replace the final fully connected layer to match your number of classes
model.fc = nn.Linear(num_ftrs, num_classes)

# --- Load Your Trained Weights ---
if not os.path.exists(model_weights_path):
    print(f"Error: Weights file not found at {model_weights_path}")
    exit()

try:
    # Load the state dictionary from your saved .pth file
    # map_location=device ensures it loads correctly regardless of where it was trained
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print(f"Successfully loaded trained weights from {model_weights_path}")
except Exception as e:
    print(f"Error loading weights: {e}")
    exit()

# --- Set Model to Evaluation Mode ---
# IMPORTANT: This disables dropout and batch normalization updates. Crucial for inference!
model.eval()

# Move the model to the specified device
model = model.to(device)

# --- Define Image Preprocessing ---
# Must be EXACTLY the same as the validation/test transforms used during training
# (Typically, no data augmentation like flips/rotations here)
preprocess = transforms.Compose([
    transforms.Resize(img_size + 32), # Or maybe just Resize(img_size)? Check your training code if unsure.
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Use same ImageNet means/stds
])

# --- Load and Predict on a New Image ---
# IMPORTANT: Update this path to the image you want to classify
image_path = '../dataset/Testing/glioma/Te-gl_0010.jpg'

if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit()

try:
    # Load the image using Pillow
    img = Image.open(image_path).convert('RGB') # Ensure image is in RGB format

    # Apply the preprocessing steps
    input_tensor = preprocess(img)

    # Add a batch dimension (model expects batches) -> [C, H, W] to [1, C, H, W]
    input_batch = input_tensor.unsqueeze(0)

    # Move the input batch to the device
    input_batch = input_batch.to(device)

    # --- Perform Inference ---
    # Use torch.no_grad() to disable gradient calculations (saves memory and computation)
    with torch.no_grad():
        output = model(input_batch) # Get model's raw output scores (logits)

    # --- Interpret the Output ---
    # Apply Softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the index of the highest probability class
    predicted_idx = torch.argmax(probabilities).item()

    # You need the mapping from index back to class name.
    # This depends on how ImageFolder ordered your classes during training.
    # You should have printed this in Cell 4 (imagefolder_class_to_idx).
    # Example (replace with your actual mapping):
    idx_to_class = {
         0: 'glioma', # Replace with YOUR actual index-to-class mapping
         1: 'meningioma',
         2: 'no tumor',
         3: 'pituitary tumor'
         # Make sure this matches the output from Cell 4 in your notebook!
    }
    # Or better, save the `train_dataset.class_to_idx` from Kaggle and load it here.

    predicted_class_name = idx_to_class.get(predicted_idx, "Unknown Index")

    print(f"\nPrediction for: {image_path}")
    print(f"Predicted Class Index: {predicted_idx}")
    print(f"Predicted Class Name: {predicted_class_name}")
    print(f"Confidence: {probabilities[predicted_idx].item():.4f}")

    # Optional: Print probabilities for all classes
    # print("Probabilities per class:")
    # for i, class_name in idx_to_class.items():
    #    print(f"  {class_name}: {probabilities[i].item():.4f}")

    # Optional: Map to your severity ranking
    severity_ranking = {'no tumor': 0, 'pituitary tumor': 1, 'meningioma': 2, 'glioma': 3}
    predicted_severity = severity_ranking.get(predicted_class_name, -1)
    print(f"Predicted Severity Level (0=lowest, 3=highest): {predicted_severity}")


except Exception as e:
    print(f"An error occurred during prediction: {e}")