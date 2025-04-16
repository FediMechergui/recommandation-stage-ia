from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from app.models.recommendation_engine import RecommendationEngine
from app.utils.data_processor import load_data, preprocess_data
from app.api.routes import api_blueprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
            static_folder='app/static',
            template_folder='app/templates')

# Enable CORS
CORS(app)

# Load configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_key_for_recommendation_system')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///internships.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Register blueprints
app.register_blueprint(api_blueprint, url_prefix='/api')

# Initialize recommendation engine
recommendation_engine = None

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/profile')
def profile():
    """Render the profile creation page"""
    return render_template('profile.html')

@app.route('/recommendations')
def recommendations():
    """Render the recommendations page"""
    try:
        return render_template('recommendations.html')
    except Exception as e:
        import traceback
        print("\n--- ERROR IN /recommendations ROUTE ---")
        traceback.print_exc()
        print("--- END ERROR ---\n")
        return f"<pre>{traceback.format_exc()}</pre>", 500

@app.route('/api/get_recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint to get recommendations based on user profile"""
    global recommendation_engine
    
    if recommendation_engine is None:
        return jsonify({"error": "Recommendation engine not initialized"}), 500
    
    # Get user profile data from request
    user_data = request.json
    
    # Get recommendations
    try:
        recommendations = recommendation_engine.recommend(user_data)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def initialize_app():
    """Initialize the application data and models"""
    global recommendation_engine
    
    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Initialize recommendation engine
    recommendation_engine = RecommendationEngine(processed_data)
    print("Recommendation engine initialized successfully!")

if __name__ == '__main__':
    import traceback
    try:
        # Initialize the app before running
        initialize_app()
        # Run the Flask application
        port = int(os.getenv('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print("\n--- ERROR DURING APP STARTUP ---")
        traceback.print_exc()
        print("--- END ERROR ---\n")
