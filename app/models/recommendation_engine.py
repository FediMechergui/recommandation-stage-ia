import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import os
from gensim.models import Word2Vec
from transformers import BertModel, BertTokenizer
import torch

class RecommendationEngine:
    """
    Main recommendation engine class that combines different recommendation strategies:
    - Collaborative Filtering
    - K-Nearest Neighbors (KNN)
    - NLP with Embeddings (Word2Vec, BERT)
    """
    
    def __init__(self, data, model_dir='app/models/saved'):
        """
        Initialize the recommendation engine
        
        Args:
            data (dict): Dictionary containing preprocessed data
            model_dir (str): Directory to save/load models
        """
        self.data = data
        self.model_dir = model_dir
        self.cf_model = None
        self.knn_model = None
        self.word2vec_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all recommendation models"""
        try:
            # Try to load pre-trained models
            self._load_models()
            print("Loaded pre-trained recommendation models")
        except:
            # Train new models if loading fails
            print("Training new recommendation models...")
            self._train_collaborative_filtering()
            self._train_knn()
            self._train_word2vec()
            self._load_bert()
            self._save_models()
    
    def _train_collaborative_filtering(self):
        """Train collaborative filtering model based on user interactions"""
        # Create a user-item matrix from interactions
        # For simplicity, we'll use a memory-based CF approach
        # In a real system, you might use matrix factorization or neural approaches
        
        # Example: Create a user-item matrix
        user_item_matrix = pd.pivot_table(
            self.data['interactions'], 
            values='rating', 
            index='user_id', 
            columns='internship_id',
            fill_value=0
        )
        
        # Calculate cosine similarity between users
        self.cf_model = {
            'user_item_matrix': user_item_matrix,
            'user_similarity': cosine_similarity(user_item_matrix)
        }
    
    def _train_knn(self):
        """Train KNN model based on skills and internship requirements"""
        # Extract feature vectors for internships
        internship_features = self.data['internship_vectors']
        
        # Train KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=5,
            algorithm='brute',  # Use brute-force search for cosine metric
            metric='cosine'
        )
        self.knn_model.fit(internship_features)
    
    def _train_word2vec(self):
        """Train Word2Vec model for text embeddings"""
        # Extract sentences (descriptions) from internships
        descriptions = [desc.split() for desc in self.data['internship_descriptions']]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=descriptions,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4
        )
    
    def _load_bert(self):
        """Load pre-trained BERT model for more advanced text embeddings"""
        try:
            # Load pre-trained BERT model and tokenizer
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"Error loading BERT model: {e}. Using Word2Vec only.")
    
    def _save_models(self):
        """Save trained models to disk"""
        # Save collaborative filtering model
        joblib.dump(
            self.cf_model,
            os.path.join(self.model_dir, 'cf_model.pkl')
        )
        
        # Save KNN model
        joblib.dump(
            self.knn_model,
            os.path.join(self.model_dir, 'knn_model.pkl')
        )
        
        # Save Word2Vec model
        if self.word2vec_model:
            self.word2vec_model.save(
                os.path.join(self.model_dir, 'word2vec_model.bin')
            )
    
    def _load_models(self):
        """Load trained models from disk"""
        # Load collaborative filtering model
        self.cf_model = joblib.load(
            os.path.join(self.model_dir, 'cf_model.pkl')
        )
        
        # Load KNN model
        self.knn_model = joblib.load(
            os.path.join(self.model_dir, 'knn_model.pkl')
        )
        
        # Load Word2Vec model
        self.word2vec_model = Word2Vec.load(
            os.path.join(self.model_dir, 'word2vec_model.bin')
        )
    
    def _get_text_embedding(self, text):
        """Get text embedding using Word2Vec or BERT"""
        if self.bert_model and self.bert_tokenizer:
            # Get BERT embedding
            try:
                inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            except:
                pass  # Fall back to Word2Vec if BERT fails
        
        # Get Word2Vec embedding (average of word vectors)
        if self.word2vec_model:
            words = text.lower().split()
            valid_words = [word for word in words if word in self.word2vec_model.wv]
            if valid_words:
                return np.mean([self.word2vec_model.wv[word] for word in valid_words], axis=0)
        
        # Return zero vector if all else fails
        return np.zeros(100)
    
    def recommend(self, user_profile, n_recommendations=5):
        """
        Generate internship recommendations based on user profile
        
        Args:
            user_profile (dict): User profile containing skills, education, experience
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended internships
        """
        # Extract user features
        user_skills = user_profile.get('skills', [])
        user_education = user_profile.get('education', '')
        user_experience = user_profile.get('experience', '')
        
        # ===== Approach 1: Content-Based using KNN =====
        # Create user feature vector
        user_vector = self._create_user_vector(user_skills, user_education, user_experience)
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors([user_vector], n_neighbors=n_recommendations)
        knn_recommendations = [self.data['internships'][i] for i in indices[0]]
        
        # ===== Approach 2: Collaborative Filtering =====
        # If user has previous interactions, use CF
        cf_recommendations = []
        if 'user_id' in user_profile and self.cf_model:
            user_id = user_profile['user_id']
            if user_id in self.cf_model['user_item_matrix'].index:
                # Find similar users
                user_idx = list(self.cf_model['user_item_matrix'].index).index(user_id)
                similar_users = np.argsort(self.cf_model['user_similarity'][user_idx])[::-1][1:6]  # Top 5 similar users
                
                # Get internships that similar users liked but the current user hasn't interacted with
                user_internships = set(self.cf_model['user_item_matrix'].columns[
                    self.cf_model['user_item_matrix'].iloc[user_idx] > 0
                ])
                
                for sim_user_idx in similar_users:
                    sim_user_id = self.cf_model['user_item_matrix'].index[sim_user_idx]
                    sim_user_internships = set(self.cf_model['user_item_matrix'].columns[
                        self.cf_model['user_item_matrix'].loc[sim_user_id] > 0
                    ])
                    
                    # Add internships liked by similar user but not seen by current user
                    for internship_id in sim_user_internships - user_internships:
                        if internship_id in self.data['internships_by_id']:
                            cf_recommendations.append(self.data['internships_by_id'][internship_id])
                            
                            if len(cf_recommendations) >= n_recommendations:
                                break
                    
                    if len(cf_recommendations) >= n_recommendations:
                        break
        
        # ===== Combine recommendations =====
        # Prioritize CF recommendations if available
        if cf_recommendations:
            # Combine KNN and CF recommendations
            combined_recommendations = []
            for i in range(min(n_recommendations, max(len(knn_recommendations), len(cf_recommendations)))):
                if i < len(cf_recommendations):
                    combined_recommendations.append(cf_recommendations[i])
                if i < len(knn_recommendations) and knn_recommendations[i] not in combined_recommendations:
                    combined_recommendations.append(knn_recommendations[i])
            
            return combined_recommendations[:n_recommendations]
        else:
            # If no CF recommendations, return KNN recommendations
            return knn_recommendations
    
    def _create_user_vector(self, skills, education, experience):
        """
        Create a feature vector for a user based on their profile
        
        Args:
            skills (list): List of user skills
            education (str): User's educational background
            experience (str): User's professional experience
            
        Returns:
            np.ndarray: Feature vector representing the user
        """
        # Initialize vector with zeros
        vector_size = self.data['internship_vectors'].shape[1]
        user_vector = np.zeros(vector_size)
        
        # Add skill-based features
        for skill in skills:
            if skill in self.data['skill_indices']:
                skill_idx = self.data['skill_indices'][skill]
                user_vector[skill_idx] = 1
        
        # Add text-based features using embeddings
        education_text = ' '.join(education.split())
        experience_text = ' '.join(experience.split())
        profile_text = ' '.join([' '.join(skills), education_text, experience_text])
        
        text_embedding = self._get_text_embedding(profile_text)
        
        # Combine skill-based and text-based features
        # Here we're assuming the first part of the vector is for skills
        # and the latter part is for text embeddings
        embedding_size = min(len(text_embedding), vector_size - len(self.data['skill_indices']))
        user_vector[len(self.data['skill_indices']):len(self.data['skill_indices'])+embedding_size] = text_embedding[:embedding_size]
        
        return user_vector
