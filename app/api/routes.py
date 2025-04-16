from flask import Blueprint, request, jsonify
import pandas as pd
import os
import json
from datetime import datetime

# Create API blueprint
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/profile', methods=['POST'])
def create_profile():
    """Create or update user profile"""
    try:
        # Get profile data from request
        profile_data = request.json
        
        # Validate required fields
        required_fields = ['name', 'skills', 'education']
        for field in required_fields:
            if field not in profile_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate user ID if not provided
        if 'user_id' not in profile_data:
            from uuid import uuid4
            profile_data['user_id'] = str(uuid4())
        
        # Save profile to database or file
        _save_profile(profile_data)
        
        return jsonify({"success": True, "user_id": profile_data['user_id']})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/profile/<user_id>', methods=['GET'])
def get_profile(user_id):
    """Get user profile by ID"""
    try:
        # Load profile from database or file
        profile = _load_profile(user_id)
        
        if profile:
            return jsonify(profile)
        else:
            return jsonify({"error": "Profile not found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/internships', methods=['GET'])
def get_internships():
    """Get all available internships"""
    try:
        # Load internships data
        internships_file = os.path.join('app', 'data', 'internships.csv')
        
        if not os.path.exists(internships_file):
            return jsonify({"error": "Internships data not found"}), 404
        
        internships_df = pd.read_csv(internships_file)
        internships = internships_df.to_dict('records')
        
        return jsonify({"internships": internships})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/internship/<int:internship_id>', methods=['GET'])
def get_internship(internship_id):
    """Get internship details by ID"""
    try:
        # Load internships data
        internships_file = os.path.join('app', 'data', 'internships.csv')
        
        if not os.path.exists(internships_file):
            return jsonify({"error": "Internships data not found"}), 404
        
        internships_df = pd.read_csv(internships_file)
        internship = internships_df[internships_df['id'] == internship_id].to_dict('records')
        
        if internship:
            return jsonify(internship[0])
        else:
            return jsonify({"error": "Internship not found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/feedback', methods=['POST'])
def add_feedback():
    """Add user feedback for a recommendation"""
    try:
        # Get feedback data from request
        feedback_data = request.json
        
        # Validate required fields
        required_fields = ['user_id', 'internship_id', 'rating']
        for field in required_fields:
            if field not in feedback_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Add timestamp
        feedback_data['timestamp'] = datetime.now().timestamp()
        
        # Save feedback to interactions file
        interactions_file = os.path.join('app', 'data', 'user_interactions.csv')
        
        # Load existing interactions or create new dataframe
        if os.path.exists(interactions_file):
            interactions_df = pd.read_csv(interactions_file)
        else:
            interactions_df = pd.DataFrame(columns=['user_id', 'internship_id', 'rating', 'timestamp'])
        
        # Add new interaction
        interactions_df = pd.concat([
            interactions_df, 
            pd.DataFrame([feedback_data])
        ], ignore_index=True)
        
        # Save updated interactions
        interactions_df.to_csv(interactions_file, index=False)
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/skills', methods=['GET'])
def get_skills():
    """Get all available skills"""
    try:
        # Load skills data
        skills_file = os.path.join('app', 'data', 'skills.csv')
        
        if not os.path.exists(skills_file):
            return jsonify({"error": "Skills data not found"}), 404
        
        skills_df = pd.read_csv(skills_file)
        skills = skills_df.to_dict('records')
        
        return jsonify({"skills": skills})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _save_profile(profile_data):
    """Save user profile to database or file system"""
    # Create profiles directory if it doesn't exist
    profiles_dir = os.path.join('app', 'data', 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)
    
    # Save profile to JSON file
    profile_file = os.path.join(profiles_dir, f"{profile_data['user_id']}.json")
    
    with open(profile_file, 'w') as f:
        json.dump(profile_data, f)

def _load_profile(user_id):
    """Load user profile from database or file system"""
    # Get profile file path
    profile_file = os.path.join('app', 'data', 'profiles', f"{user_id}.json")
    
    # Check if profile exists
    if not os.path.exists(profile_file):
        return None
    
    # Load profile from JSON file
    with open(profile_file, 'r') as f:
        profile_data = json.load(f)
    
    return profile_data
