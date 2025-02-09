import os
import time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from functools import wraps
import json
from geopy.geocoders import Nominatim
from math import radians, sin, cos, atan2, sqrt
from config.site_info import SITE_INFO
from flask_login import logout_user, LoginManager, login_user, login_required, current_user, UserMixin
from models import db, User, Analysis  # Import all models from models.py
from datetime import datetime



# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pneumonia_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/landing')
def landing():
    return render_template('landing.html', site_info=SITE_INFO)

@app.route('/', methods=['GET', 'POST'])
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('landing'))
    
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).all()
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        patient_name = request.form.get('patient_name', 'Unknown')
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Process image and get predictions
                img_array = preprocess_image(file_path)
                results = {}
                votes = []
                
                for model_name, model in models.items():
                    prediction = model.predict(img_array)[0][0]
                    pred_label = "Pneumonia" if prediction > 0.5 else "Normal"
                    results[model_name] = pred_label
                    votes.append(pred_label)
                
                final_prediction = Counter(votes).most_common(1)[0][0]
                
                # Clean up uploaded file
                os.remove(file_path)
                
                # Save analysis to database
                analysis = Analysis(
                    user_id=current_user.id,
                    patient_name=patient_name,
                    prediction=final_prediction
                )
                db.session.add(analysis)
                db.session.commit()
                
                return render_template('index.html',
                    show_upload=False,
                    patient_name=patient_name,
                    results=results,
                    final_prediction=final_prediction,
                    analyses=analyses,
                    user=current_user,
                    site_info=SITE_INFO
                )
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(request.url)
    
    return render_template('index.html', 
                         show_upload=True, 
                         analyses=analyses,
                         user=current_user, 
                         site_info=SITE_INFO)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html', site_info=SITE_INFO)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))

        user = User(name=name, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error during registration: {str(e)}', 'error')
            return redirect(url_for('register'))

    return render_template('register.html', site_info=SITE_INFO)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Configure upload folder
UPLOAD_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Directory containing models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Mapping raw model filenames (without extension) to user-friendly names.
model_name_mapping = {
    'model1': 'Basic CNN (3 Conv2D + 64 Dense)',
    'model2': 'Deep CNN (4 Conv2D + 64 Dense)',
    'model3': 'Light CNN (2 Conv2D + 64 Dense)',
    'model4': 'Enhanced CNN (3 Conv2D + 128 Dense)',
    'model5': 'Multi-Filter CNN (32-64-128 filters)',
    'model6': 'Regularized CNN (32-64-128 + Dropout)'
}

# Load all .keras models from the directory
models = {}

def load_models():
    """Load all .keras models from the models directory"""
    try:
        for file in os.listdir(MODEL_DIR):
            if file.endswith('.keras'):
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(MODEL_DIR, file)
                try:
                    models[model_name] = load_model(model_path)
                    print(f"Loaded model: {model_name}")
                except Exception as e:
                    print(f"Error loading model {file}: {e}")
    except Exception as e:
        print(f"Error accessing models directory: {e}")

def init_app():
    """Initialize the application"""
    with app.app_context():
        # Create database tables
        db.create_all()
        print("Database tables created successfully!")
        
        # Load models
        load_models()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image for the model."""
    img = image.load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to get nearby hospitals
def get_nearby_hospitals(lat, lon):
    try:
        print(f"Searching for hospitals near coordinates: {lat}, {lon}")  # Debug log
        
        # Try Nominatim first
        geolocator = Nominatim(user_agent="pneumonia_app")
        
        # Search in a larger area (20km radius approximately)
        search_query = f"""
        [out:json][timeout:25];
        (
            area[name="India"]->.searchArea;
            (
                node["amenity"="hospital"](around:20000, {lat}, {lon});
                way["amenity"="hospital"](around:20000, {lat}, {lon});
                relation["amenity"="hospital"](around:20000, {lat}, {lon});
                node["healthcare"="hospital"](around:20000, {lat}, {lon});
                way["healthcare"="hospital"](around:20000, {lat}, {lon});
                relation["healthcare"="hospital"](around:20000, {lat}, {lon});
            );
        );
        out body center;
        >;
        out skel qt;
        """
        
        # Try multiple Overpass API endpoints
        overpass_endpoints = [
            "https://overpass-api.de/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"
        ]
        
        data = None
        for endpoint in overpass_endpoints:
            try:
                print(f"Trying endpoint: {endpoint}")  # Debug log
                response = requests.post(endpoint, data=search_query, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    break
            except Exception as e:
                print(f"Failed with endpoint {endpoint}: {str(e)}")
                continue
        
        if not data:
            print("No data received from any endpoint")
            # Fallback to hardcoded major hospitals if no results
            return get_fallback_hospitals(lat, lon)
            
        hospitals = []
        elements = data.get('elements', [])
        print(f"Found {len(elements)} raw elements")  # Debug log
        
        for element in elements:
            tags = element.get('tags', {})
            name = tags.get('name')
            if not name:
                continue
                
            # Get coordinates
            if element.get('type') == 'node':
                hospital_lat = element.get('lat')
                hospital_lon = element.get('lon')
            else:
                center = element.get('center', {})
                hospital_lat = center.get('lat')
                hospital_lon = center.get('lon')
            
            if not (hospital_lat and hospital_lon):
                continue
            
            # Calculate distance
            distance = calculate_distance(lat, lon, hospital_lat, hospital_lon)
            
            # Get address
            address_parts = []
            for key in ['addr:housenumber', 'addr:street', 'addr:city', 'addr:state']:
                if tags.get(key):
                    address_parts.append(tags.get(key))
            
            address = ", ".join(address_parts) if address_parts else tags.get('addr:full', 'Address not available')
            
            # Try to get additional address info from Nominatim if address is not available
            if address == 'Address not available':
                try:
                    location = geolocator.reverse(f"{hospital_lat}, {hospital_lon}", exactly_one=True)
                    if location and location.address:
                        address = location.address
                except Exception as e:
                    print(f"Nominatim reverse geocoding failed: {e}")
            
            hospital_info = {
                'name': name,
                'address': address,
                'distance': f"{distance:.1f} km away",
                'phone': tags.get('phone', tags.get('contact:phone', 'Phone not available')),
                'emergency': tags.get('emergency') == 'yes'
            }
            hospitals.append(hospital_info)
        
        # Sort by distance
        hospitals.sort(key=lambda x: float(x['distance'].split()[0]))
        
        print(f"Processed {len(hospitals)} valid hospitals")  # Debug log
        
        if not hospitals:
            return get_fallback_hospitals(lat, lon)
            
        return hospitals[:10]
        
    except Exception as e:
        print(f"Error in get_nearby_hospitals: {e}")
        return get_fallback_hospitals(lat, lon)

def get_fallback_hospitals(lat, lon):
    """Fallback function that returns major hospitals in India"""
    try:
        major_hospitals = [
            {
                'name': 'AIIMS Delhi',
                'address': 'Sri Aurobindo Marg, Ansari Nagar, New Delhi, Delhi 110029',
                'coordinates': (28.5672, 77.2100)
            },
            {
                'name': 'Fortis Hospital',
                'address': 'Sector B-1, Vasant Kunj, New Delhi, Delhi 110070',
                'coordinates': (28.5231, 77.1571)
            },
            {
                'name': 'Apollo Hospitals',
                'address': 'Plot No 1A, Bhat GIDC Estate, Gandhinagar, Gujarat 382428',
                'coordinates': (23.1127, 72.5847)
            }
        ]
        
        hospitals = []
        for hospital in major_hospitals:
            try:
                distance = calculate_distance(lat, lon, 
                                           hospital['coordinates'][0], 
                                           hospital['coordinates'][1])
                hospitals.append({
                    'name': hospital['name'],
                    'address': hospital['address'],
                    'distance': f"{distance:.1f} km away" if distance else "Distance unavailable",
                    'phone': 'Contact hospital directory',
                    'emergency': True
                })
            except Exception as e:
                print(f"Error processing fallback hospital {hospital['name']}: {str(e)}")
                continue
        
        # Sort by distance if available
        hospitals.sort(key=lambda x: float(x['distance'].split()[0]) if 'km' in x['distance'] else float('inf'))
        return hospitals
        
    except Exception as e:
        print(f"Error in fallback hospitals: {str(e)}")
        return []

def calculate_distance(lat1, lon1, lat2, lon2):
    try:
        # Earth's radius in kilometers
        R = 6371.0
        
        # Convert coordinates to radians
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        
        # Differences in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return round(distance, 2)
    except Exception as e:
        print(f"Error calculating distance: {str(e)}")
        return None

# Update the home route to handle location data
@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Handle file upload and analysis
        if 'xray_image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['xray_image']
        patient_name = request.form.get('patient_name', 'Unknown Patient')
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Secure filename with a timestamp to avoid overwrites
                filename = secure_filename(f"{int(time.time())}_{file.filename}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                print(f"File saved at: {file_path}")

                # Preprocess image
                img_array = preprocess_image(file_path)
                print("Image preprocessed successfully.")

                # Get predictions from all models
                results = {}
                votes = []  # To store each model's vote for final prediction
                for model_name, model in models.items():
                    try:
                        prediction = model.predict(img_array)[0][0]
                        pred_label = "Pneumonia" if prediction > 0.5 else "Normal"
                        results[model_name] = pred_label
                        votes.append(pred_label)
                        print(f"{model_name} Prediction: {pred_label}")
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        results[model_name] = error_msg
                        print(f"Error predicting with {model_name}: {error_msg}")

                # Determine final prediction by majority vote
                if votes:
                    final_prediction = Counter(votes).most_common(1)[0][0]
                else:
                    final_prediction = "N/A"

                # After processing/sending the image
                try:
                    os.remove(file_path)  # Delete the image file
                    print(f"Successfully deleted {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

                return render_template(
                    'index.html', 
                    patient_name=patient_name, 
                    results=results, 
                    final_prediction=final_prediction,
                    image_filename=filename,
                    current_user=current_user,
                    site_info=SITE_INFO
                )
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(request.url)

    # GET request - just show the upload form
    return render_template('index.html', 
                         current_user=current_user,
                         site_info=SITE_INFO)

@app.route('/get_hospitals', methods=['POST'])
@login_required
def get_hospitals():
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        # Add detailed debug logging
        print(f"Debug: Received request data: {data}")
        print(f"Debug: Latitude: {latitude}, Longitude: {longitude}")
        
        if not latitude or not longitude:
            print("Error: Missing coordinates in request")
            return jsonify({"error": "Missing coordinates"}), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        try:
            lat = float(latitude)
            lon = float(longitude)
        except (ValueError, TypeError) as e:
            print(f"Error: Invalid coordinates format - {e}")
            return jsonify({"error": "Invalid coordinates format"}), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        try:
            hospitals = get_nearby_hospitals(lat, lon)
            print(f"Debug: Found {len(hospitals)} hospitals")
            
            if not hospitals:
                print("Debug: No hospitals found, using fallback")
                hospitals = get_fallback_hospitals(lat, lon)
                
            if not hospitals:
                print("Warning: No hospitals found even with fallback")
                hospitals = []
                
            response = jsonify(hospitals)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response
            
        except Exception as e:
            print(f"Error in hospital search: {str(e)}")
            return jsonify({"error": "Error searching hospitals"}), 500, {'Content-Type': 'application/json; charset=utf-8'}
            
    except Exception as e:
        print(f"Critical error in get_hospitals: {str(e)}")
        return jsonify({"error": "Server error"}), 500, {'Content-Type': 'application/json; charset=utf-8'}

# Add this to ensure all responses have the correct charset
@app.after_request
def add_header(response):
    if 'Content-Type' in response.headers and 'application/json' in response.headers['Content-Type']:
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# Add this to your existing routes
@app.context_processor
def inject_site_info():
    """Make site info available to all templates"""
    return dict(site_info=SITE_INFO)

if __name__ == '__main__':
    init_app()  # Initialize database and load models
    app.run(debug=True)
