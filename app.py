import os
# <<< ADDED/MODIFIED START >>>
# Imports for model, image processing, etc.
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image # Library to load images
import io # For processing image in memory
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
model_weights_path = './ml_training/resnet_brain_tumor_weights.pth'
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
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)), # Resize directly
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("Preprocessing transform created.")

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

        img_bytes = None
        # --- Image Processing and Prediction ---
        try:
            print(f"Processing file: {file.filename}") # Log filename
            # Read image file into memory
            img_bytes = file.read()
            if not img_bytes:
                 flash('Uploaded file appears to be empty.')
                 return redirect(url_for('index'))
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB') # Ensure RGB

            # Preprocess the image
            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dim and move to device

            # Perform Inference
            print("Running model inference...")
            with torch.no_grad():
                output = model(input_batch) # Get model's raw output scores (logits)

            # Interpret results
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx_tensor = torch.max(probabilities, 0)
            predicted_idx = predicted_idx_tensor.item() # Get Python integer index

            predicted_class_name = idx_to_class.get(predicted_idx, "Unknown Index")
            predicted_severity = severity_ranking.get(predicted_class_name, -1)
            confidence_percent = confidence.item() * 100 # Convert confidence to percentage

            print(f"Prediction: {predicted_class_name}, Confidence: {confidence_percent:.2f}%")

            file_extension = original_filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            save_path = os.path.join(app.config['HISTORY_UPLOAD_FOLDER'], unique_filename)

            try:
                # Save the image bytes we read earlier
                with open(save_path, 'wb') as f:
                    f.write(img_bytes)
                print(f"Image saved to: {save_path}")

                # Create history record using the imported model
                new_record = AnalysisHistory(
                    user_id=current_user.id,
                    original_filename=original_filename,
                    stored_filename=unique_filename,
                    predicted_class=predicted_class_name,
                    confidence=confidence_percent,
                    severity_level=predicted_severity
                )
                db.session.add(new_record)
                db.session.commit()
                print("Analysis record saved to database.")
                flash('Analysis successful and saved to history.', 'success') # Add success message

            except Exception as save_err:
                print(f"Error saving file or DB record: {save_err}")
                db.session.rollback()
                flash('Analysis complete, but failed to save history record.', 'error')
            # <<< ADDED/MODIFIED END >>>

            # Render results on the index page with correct parameters
            return render_template('index.html',
                                username=current_user.username,
                                prediction=predicted_class_name,
                                filename=file.filename,
                                confidence=f"{confidence_percent:.2f}", # Format confidence with 2 decimal places
                                severity=predicted_severity)

        except Exception as e:
            print(f"Error during analysis for file {file.filename}: {e}") # Log the error
            flash(f'An error occurred during image analysis. Please try again or contact support.')
            # Render index page but indicate an error occurred
            return render_template('index.html',
                                username=current_user.username)
        # <<< ADDED/MODIFIED END >>>

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