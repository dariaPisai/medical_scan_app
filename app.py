import os
# <<< ADDED/MODIFIED START >>>
# <<< ADDED/MODIFIED START >>>
# Imports for model, image processing, etc.
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image # Library to load images
import io # For processing image in memory
# --- Add these ---
import cv2         # OpenCV for image processing used in crop_img
import numpy as np # Already imported but ensure it's available
import imutils     # Required by crop_img (pip install imutils)
# Imports for model, image processing, etc.
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image # Library to load images
import io # For processing image in memory
import pydicom # For reading DICOM
from pydicom.pixel_data_handlers.util import apply_voi_lut # More robust windowing
import numpy as np
from PIL import Image
import io
from flask import send_file # For sending file for download
# <<< ADDED/MODIFIED END >>>

from werkzeug.utils import secure_filename # To secure original filenames
import uuid # To generate unique filenames for storage
from datetime import datetime # For timestamping history records
from flask import send_from_directory # To serve saved images

from flask import Flask, render_template, request, redirect, url_for, flash # Added flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, current_user
from flask_migrate import Migrate

# Variabile globale, nu au nevoie de contextul aplicatiei
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()

# Configuram login manager
login_manager.login_view = 'auth.login' # Numele rutei pentru login page
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):

    try:
        from db_models import User
    except ImportError:
        print("ERROR: Could not import User from db_models. Make sure db_models.py exists and is importable.")
        # Handle appropriately, maybe raise error or return None
        return None
    return User.query.get(int(user_id))

# <<< ADDED/MODIFIED START >>>
# --- Configuration & Model Loading ---
# Load the model only ONCE when the app starts

num_classes = 4
img_size = 224
# Assuming app.py is in the root 'medical_scan_app' folder
# Adjust if app.py is located elsewhere relative to the weights file
model_weights_path = './ml_training/resnet50_tumor_classifier_best_val_acc.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Attempting to use device: {device}") # Log device

# Define the model architecture
model = models.resnet50(weights=None) # Load architecture only
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the trained weights
if os.path.exists(model_weights_path):
    try:
        # Load weights, ensuring correct mapping to device
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Successfully loaded trained weights from {model_weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}. Model will have random weights.")
        # Depending on your needs, you might want to prevent the app from running
        # if weights fail to load, e.g., raise SystemExit("Failed to load model weights.")
else:
    print(f"Warning: Weights file not found at {model_weights_path}. Model initialized with random weights.")
    # Consider adding a stronger warning or preventing startup

# Set model to evaluation mode and move to device
model.eval()
model = model.to(device)
print(f"Model ready on device: {device}")

# --- Image Preprocessing Transform ---
# (Must match the validation/inference transform used before)
preprocess_after_crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("Preprocessing transform (ToTensor, Normalize) created.")

def crop_img(img, target_size=img_size): # Pass target_size
    """
    Finds the presumed tumor area based on contours, crops, and resizes the image.
    Handles grayscale, color, and alpha channel images.
    Returns None if cropping or resizing fails critically.
    Input: NumPy array (BGR format preferred).
    Output: NumPy array (BGR format, resized) or None.
    """
    # Input validation
    if img is None:
        print("Warning: crop_img received a None image.")
        return None
    if not isinstance(img, np.ndarray) or img.ndim < 2:
         print(f"Warning: crop_img received invalid image data type/dims: {type(img)}, ndim={img.ndim if isinstance(img, np.ndarray) else 'N/A'}")
         return None
    if img.shape[0] <= 0 or img.shape[1] <= 0:
        print(f"Warning: crop_img received image with invalid dimensions: {img.shape}")
        return None

    # Ensure image is BGR
    if len(img.shape) == 2 or img.shape[2] == 1: # Grayscale
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4: # BGRA
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 3: # BGR
        img_bgr = img
    else:
        print(f"Warning: Unsupported number of channels ({img.shape[2]}) in image. Cannot process.")
        try:
            # Attempt resize if valid shape
            if img.shape[0] > 0 and img.shape[1] > 0:
                 print("  Attempting direct resize as fallback.")
                 return cv2.resize(img, (target_size, target_size))
            else:
                 return None
        except Exception as e:
            print(f"  Fallback resize also failed: {e}")
            return None


    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Fallback: If no contours, resize the original BGR image
    if not cnts:
        print("Warning: No contours found. Resizing original image.")
        try:
            return cv2.resize(img_bgr, (target_size, target_size))
        except Exception as e:
             print(f"  Fallback resize failed for no-contour image: {e}")
             return None # Indicate failure

    # Find the largest contour and its bounding box
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop using bounding box coordinates (with bounds checking)
    ADD_PIXELS = 0 # Optional buffer
    top = max(0, extTop[1] - ADD_PIXELS)
    bottom = min(img_bgr.shape[0], extBot[1] + ADD_PIXELS)
    left = max(0, extLeft[0] - ADD_PIXELS)
    right = min(img_bgr.shape[1], extRight[0] + ADD_PIXELS)

    # Validate crop dimensions
    if top >= bottom or left >= right:
        print(f"Warning: Invalid crop dimensions calculated ({top}:{bottom}, {left}:{right}). Resizing original image.")
        try:
            return cv2.resize(img_bgr, (target_size, target_size))
        except Exception as e:
             print(f"  Fallback resize failed for invalid crop dims: {e}")
             return None # Indicate failure

    cropped_img = img_bgr[top:bottom, left:right].copy()

    # Final resize of the cropped image
    try:
        # *** IMPORTANT: crop_img now returns the resized image ***
        resized_img = cv2.resize(cropped_img, (target_size, target_size))
        return resized_img
    except Exception as e:
        print(f"Error resizing cropped image: {e}. Image shape was {cropped_img.shape}")
        # Attempt to resize original as last resort
        try:
             print("  Attempting resize of original image instead.")
             return cv2.resize(img_bgr, (target_size, target_size))
        except Exception as e_orig:
             print(f"    Resize of original image also failed: {e_orig}")
             return None

# --- Class Mappings (Crucial!) ---
# This MUST match the mapping used by ImageFolder during training
# Check Cell 4 output in your Kaggle notebook
# Example (REPLACE WITH YOUR ACTUAL MAPPING from Kaggle Cell 4 output):
idx_to_class = {
     0: 'glioma',
     1: 'meningioma',
     2: 'no tumor',
     3: 'pituitary tumor'
}
severity_ranking = {'no tumor': 0, 'pituitary tumor': 1, 'meningioma': 2, 'glioma': 3}
print(f"Using idx_to_class mapping: {idx_to_class}")

# --- Allowed File Extensions ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# <<< ADDED/MODIFIED END >>>

def apply_dicom_windowing(dicom_dataset):
    """
    Applies VOI LUT or Window Center/Width to DICOM pixel data
    and converts to an 8-bit NumPy array suitable for display.
    """
    # Use pydicom's apply_voi_lut for robust windowing if VOI LUT Sequence or WC/WW is present
    # It handles Rescale Slope/Intercept automatically
    try:
        # This function handles different scenarios (VOI LUT, WC/WW)
        # It typically returns data ready for display, possibly > 8-bit initially
        windowed_data = apply_voi_lut(dicom_dataset.pixel_array, dicom_dataset)
        print("Applied VOI LUT or Window Center/Width.")
    except Exception as e:
        # Fallback if apply_voi_lut fails or tags are missing
        print(f"VOI LUT/Windowing failed ({e}), applying simple scaling fallback.")
        pixels = dicom_dataset.pixel_array.astype(float)
        # Simple min-max scaling to 0-255
        pixels = pixels - np.min(pixels)
        pixels = pixels / (np.max(pixels) + 1e-8) # Avoid division by zero
        windowed_data = (pixels * 255.0)

    # Ensure data is scaled correctly to 8-bit (0-255)
    # apply_voi_lut might return values outside this range depending on config
    pixels_8bit = np.clip(windowed_data, 0, np.max(windowed_data)) # Clip just in case
    if np.max(pixels_8bit) > 0: # Avoid division by zero if image is black
         pixels_8bit = (pixels_8bit / np.max(pixels_8bit)) * 255.0
    pixels_8bit = pixels_8bit.astype(np.uint8)
    return pixels_8bit
# <<< ADDED END: Helper function for DICOM Windowing >>>

def create_app():
    app = Flask(__name__, instance_relative_config=True) # flask se uita pt config in folderul instance

    # Use a strong, unique secret key, potentially from environment variables
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-random-string123-change-me') # Changed default
    db_path = os.path.join(app.instance_path, 'database.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # Removed UPLOAD_FOLDER config as we process in memory now

    history_upload_folder = os.path.join(app.instance_path, 'history_uploads')
    app.config['HISTORY_UPLOAD_FOLDER'] = history_upload_folder

    try:
        # Ensure the instance folder exists
        os.makedirs(app.instance_path, exist_ok=True) # Added exist_ok=True
        os.makedirs(app.config['HISTORY_UPLOAD_FOLDER'], exist_ok=True) # Create history folder
        print(f"Ensured history upload folder exists: {app.config['HISTORY_UPLOAD_FOLDER']}")
    except OSError as e:
        print(f"Error creating instance folder: {e}") # Log error
        pass # Or handle more gracefully

    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    try:
        # Ensure models are imported within the app context or accessible globally
        # If db_models.py imports 'db' from this file, import it later or restructure.
        from db_models import AnalysisHistory # Import the history model
        print("AnalysisHistory model imported successfully.")
    except ImportError as e:
        print(f"ERROR: Could not import AnalysisHistory from db_models: {e}")
        # Handle this error - perhaps disable history features if model isn't found
    # <<< ADDED/MODIFIED END >>>

    # Import and register blueprints
    # Make sure 'auth' blueprint is defined correctly in auth.py
    try:
        from auth import auth_bp
        app.register_blueprint(auth_bp, url_prefix='/auth')
    except ImportError:
        print("Warning: Could not import or register 'auth' blueprint.")


    @app.route('/')
    @login_required
    def index():
        # Render the main page, potentially passing flashed messages if needed later
        return render_template('index.html', username=current_user.username)

    # <<< ADDED/MODIFIED START >>>
    # ... keep existing code ...

    @app.route('/analyze', methods=['POST'])
    @login_required
    def analyze():
        # --- File Handling ---
        if 'scan_file' not in request.files:
            flash('No file part provided.')
            return redirect(url_for('index'))

        file = request.files['scan_file']

        if file.filename == '':
            flash('No file selected.')
            return redirect(url_for('index'))

        original_filename = secure_filename(file.filename)

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PNG, JPG, or JPEG.')
            return redirect(url_for('index'))

        img_bytes_for_saving = None # Store original bytes for saving later
        # --- Image Processing and Prediction ---
        try:
            print(f"Processing file: {file.filename}")
            # Read image file into memory
            img_bytes = file.read()
            if not img_bytes:
                 flash('Uploaded file appears to be empty.')
                 return redirect(url_for('index'))
            img_bytes_for_saving = img_bytes # Keep original bytes

            # <<< MODIFIED START: Apply crop_img >>>
            # 1. Decode image bytes into an OpenCV NumPy array (BGR)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Use cv2.IMREAD_COLOR for BGR

            if img_cv is None:
                 flash('Could not decode image file. It might be corrupted or an unsupported format.')
                 return redirect(url_for('index'))

            # 2. Apply the crop_img function
            print("Applying crop_img preprocessing...")
            processed_cv_img = crop_img(img_cv, target_size=img_size) # crop_img now resizes

            if processed_cv_img is None:
                 flash('Image preprocessing (cropping/resizing) failed. Please check image format or content.')
                 return redirect(url_for('index'))

            # 3. Convert the processed NumPy array (BGR) back to a PIL Image (RGB)
            #    OpenCV uses BGR, PIL uses RGB
            img_pil = Image.fromarray(cv2.cvtColor(processed_cv_img, cv2.COLOR_BGR2RGB))
            # <<< MODIFIED END >>>

            # 4. Apply the remaining transforms (ToTensor, Normalize)
            input_tensor = preprocess_after_crop(img_pil) # Use the updated transform
            input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dim and move to device

            # --- Perform Inference ---
            print("Running model inference...")
            with torch.no_grad():
                output = model(input_batch)

            # --- Interpret results ---
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx_tensor = torch.max(probabilities, 0)
            predicted_idx = predicted_idx_tensor.item()

            predicted_class_name = idx_to_class.get(predicted_idx, "Unknown Index")
            predicted_severity = severity_ranking.get(predicted_class_name, -1)
            confidence_percent = confidence.item() * 100

            print(f"Prediction: {predicted_class_name}, Confidence: {confidence_percent:.2f}%")

            # --- Save original image and history record ---
            file_extension = original_filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            save_path = os.path.join(app.config['HISTORY_UPLOAD_FOLDER'], unique_filename)

            try:
                # Save the *original* image bytes we stored earlier
                with open(save_path, 'wb') as f:
                    f.write(img_bytes_for_saving) # Save the original file content
                print(f"Original image saved to: {save_path}")

                # Create history record
                new_record = AnalysisHistory(
                    user_id=current_user.id,
                    original_filename=original_filename,
                    stored_filename=unique_filename,
                    predicted_class=predicted_class_name,
                    confidence=confidence_percent, # This already has *100 applied
                    severity_level=predicted_severity
                )
                db.session.add(new_record)
                db.session.commit()
                print("Analysis record saved to database.")
                flash('Analysis successful and saved to history.', 'success')

            except Exception as save_err:
                print(f"Error saving file or DB record: {save_err}")
                db.session.rollback()
                flash('Analysis complete, but failed to save history record.', 'error')

            # --- Render results ---
            return render_template('index.html',
                                username=current_user.username,
                                prediction=predicted_class_name,
                                filename=original_filename, # Show original filename
                                confidence=f"{confidence_percent:.2f}",
                                severity=predicted_severity)

        except ImportError:
            # Specific error if AnalysisHistory couldn't be imported
            print("ERROR: AnalysisHistory model not available for saving.")
            flash('Analysis complete, but history feature is unavailable.', 'warning')
             # Still show results even if history fails due to import error
            return render_template('index.html',
                                username=current_user.username,
                                prediction=predicted_class_name,
                                filename=original_filename,
                                confidence=f"{confidence_percent:.2f}",
                                severity=predicted_severity)
        except Exception as e:
            print(f"Error during analysis for file {original_filename}: {e}") # Log the error
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            flash(f'An error occurred during image analysis. Please check the logs or contact support.')
            # Render index page but indicate an error occurred
            return render_template('index.html',
                                username=current_user.username)
        

    # <<< ADDED/MODIFIED START >>>
    # --- Add History Route ---
    @app.route('/history')
    @login_required
    def history():
        try:
            # Fetch history records for the current user, newest first
            user_history = AnalysisHistory.query.filter_by(user_id=current_user.id)\
                                                .order_by(AnalysisHistory.timestamp.desc())\
                                                .all()
            return render_template('history.html',
                                   username=current_user.username,
                                   history=user_history)
        except Exception as e:
            print(f"Error fetching history for user {current_user.id}: {e}")
            flash("Could not retrieve analysis history.", "error")
            return render_template('history.html', username=current_user.username, history=[])

    # --- Add Route to Serve History Images ---
    @app.route('/history_images/<path:filename>')
    @login_required
    def get_history_image(filename):
        # Security check: Ensure the requested file belongs to the logged-in user
        record = AnalysisHistory.query.filter_by(user_id=current_user.id, stored_filename=filename).first_or_404()
        # Using first_or_404() simplifies - Flask handles the 404 if no record found
        # Note: This doesn't prevent timing attacks but is common practice.

        try:
            # Serve the file from the configured history upload folder
            print(f"Serving image: {filename} from {app.config['HISTORY_UPLOAD_FOLDER']}")
            return send_from_directory(
                app.config['HISTORY_UPLOAD_FOLDER'],
                filename,
                as_attachment=False # Display inline
            )
        except FileNotFoundError:
             print(f"History image file physically not found: {filename}")
             # Return 404 explicitly if file missing on disk but DB record exists
             from flask import abort
             abort(404)
        except Exception as e:
             print(f"Error serving image {filename}: {e}")
             from flask import abort
             abort(500) # Internal server error

    
    @app.route('/dicom_converter')
    @login_required # Keep consistent with other app sections
    def dicom_converter_form():
        """Renders the upload form for DICOM conversion."""
        return render_template('dicom_converter.html', username=current_user.username)

    @app.route('/convert_dicom', methods=['POST'])
    @login_required
    def convert_dicom():
        """Handles DICOM file upload, converts to JPG, and sends for download."""
        if 'dicom_file' not in request.files:
            flash('No file part provided.', 'error')
            return redirect(url_for('dicom_converter_form'))

        file = request.files['dicom_file']

        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('dicom_converter_form'))

        # Check file extension server-side
        if not (file.filename.lower().endswith('.dcm') or file.filename.lower().endswith('.dicom')):
            flash('Invalid file type. Please upload a .dcm or .dicom file.', 'error')
            return redirect(url_for('dicom_converter_form'))

        # Create a safe output filename based on the original name
        original_filename_base = os.path.splitext(secure_filename(file.filename))[0]
        output_filename = f"{original_filename_base}.jpg"

        try:
            # Read DICOM directly from the file stream
            dicom_dataset = pydicom.dcmread(file.stream)

            # --- Handle Pixel Data ---
            if 'PixelData' not in dicom_dataset:
                 flash('DICOM file does not contain pixel data.', 'error')
                 return redirect(url_for('dicom_converter_form'))

            pixel_array = dicom_dataset.pixel_array

            # --- Handle Multi-frame DICOM (Select Middle Frame) ---
            if pixel_array.ndim > 2:
                print(f"Multi-frame DICOM detected (shape: {pixel_array.shape}). Selecting middle frame.")
                middle_frame_index = pixel_array.shape[0] // 2
                single_frame_pixels = pixel_array[middle_frame_index]
                flash(f'Multi-frame DICOM detected. Converted middle frame ({middle_frame_index+1}/{pixel_array.shape[0]}).', 'info')
            elif pixel_array.ndim == 2:
                single_frame_pixels = pixel_array
            else:
                flash('Unsupported pixel data dimensions.', 'error')
                return redirect(url_for('dicom_converter_form'))

            # --- Apply Windowing and Convert to 8-bit ---
            pixels_8bit = apply_dicom_windowing(dicom_dataset)
            # Overwrite single_frame_pixels with windowed version if it was multiframe originally
            if pixel_array.ndim > 2:
                 single_frame_pixels = pixels_8bit[middle_frame_index] 
            else:
                 single_frame_pixels = pixels_8bit

            # --- Create PIL Image and Save as JPG in Memory ---
            # Ensure the array is contiguous for PIL
            if not single_frame_pixels.flags['C_CONTIGUOUS']:
                single_frame_pixels = np.ascontiguousarray(single_frame_pixels)

            img = Image.fromarray(single_frame_pixels).convert('L') # Convert to Grayscale for JPG

            img_io = io.BytesIO() # Create in-memory stream
            img.save(img_io, 'JPEG', quality=90) # Higher quality (90 instead of 85)
            img_io.seek(0) # Rewind the stream to the beginning

            # --- Send the JPG file for download with proper filename ---
            return send_file(
                img_io,
                mimetype='image/jpeg',
                as_attachment=True, # Trigger download dialog
                download_name=output_filename # Set the download filename
            )

        except Exception as e:
            print(f"Error during DICOM conversion: {e}")
            flash(f'An error occurred during conversion: {str(e)}', 'error')
            return redirect(url_for('dicom_converter_form'))

    # <<< ADDED END: DICOM Converter Routes >>>


    # <<< ADDED/MODIFIED END >>>


    return app

# --- Application Execution ---
# It's common practice to create the app instance outside this function
# if you need to run it directly (e.g., python app.py)
# or use a WSGI server like Gunicorn/Waitress in production.

# Example for direct execution (for development):
# if __name__ == '__main__':
#     app = create_app()
#     app.run(debug=True) # Use debug=False in production!