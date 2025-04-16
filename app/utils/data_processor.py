import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def load_data(data_dir='app/data'):
    """
    Load internship and skills data from files
    
    Args:
        data_dir (str): Directory containing data files
    
    Returns:
        tuple: (internships_data, skills_data)
    """
    # Check if data files exist
    internships_file = os.path.join(data_dir, 'internships.csv')
    skills_file = os.path.join(data_dir, 'skills.csv')
    interactions_file = os.path.join(data_dir, 'user_interactions.csv')
    
    # If files don't exist, create sample data
    if not (os.path.exists(internships_file) and os.path.exists(skills_file)):
        print("Data files not found. Creating sample data...")
        _create_sample_data(data_dir)
    
    # Load data
    internships_data = pd.read_csv(internships_file)
    skills_data = pd.read_csv(skills_file)
    
    # Load interactions if they exist
    interactions_data = None
    if os.path.exists(interactions_file):
        interactions_data = pd.read_csv(interactions_file)
    else:
        # Create empty interactions dataframe
        interactions_data = pd.DataFrame(columns=['user_id', 'internship_id', 'rating', 'timestamp'])
    
    return {
        'internships': internships_data,
        'skills': skills_data,
        'interactions': interactions_data
    }

def preprocess_data(data):
    """
    Preprocess data for recommendation engine
    
    Args:
        data (dict): Dictionary containing dataframes
    
    Returns:
        dict: Processed data ready for the recommendation engine
    """
    internships_df = data['internships']
    skills_df = data['skills']
    interactions_df = data['interactions']
    
    # Create internships dictionary for easy lookup
    internships = []
    internships_by_id = {}
    for _, row in internships_df.iterrows():
        internship = {
            'id': row['id'],
            'title': row['title'],
            'company': row['company'],
            'description': row['description'],
            'required_skills': _parse_skills(row['required_skills']),
            'location': row['location'],
            'duration': row['duration'],
            'salary': row['salary'] if 'salary' in row else None
        }
        internships.append(internship)
        internships_by_id[row['id']] = internship
    
    # Create skill indices for vector representation
    all_skills = set()
    for internship in internships:
        all_skills.update(internship['required_skills'])
    
    skill_indices = {skill: i for i, skill in enumerate(sorted(all_skills))}
    
    # Create feature vectors for internships
    internship_vectors = np.zeros((len(internships), len(skill_indices) + 100))  # 100 for text embeddings
    internship_descriptions = []
    
    for i, internship in enumerate(internships):
        # Add skill-based features
        for skill in internship['required_skills']:
            skill_idx = skill_indices[skill]
            internship_vectors[i, skill_idx] = 1
        
        # Store description for text processing
        internship_descriptions.append(internship['description'])
    
    # Process text descriptions
    vectorizer = TfidfVectorizer(max_features=100)
    try:
        text_features = vectorizer.fit_transform(internship_descriptions).toarray()
        # Add text features to vectors
        internship_vectors[:, len(skill_indices):len(skill_indices)+100] = text_features
    except:
        # If text processing fails, just leave zeros
        pass
    
    return {
        'internships': internships,
        'internships_by_id': internships_by_id,
        'skills': all_skills,
        'skill_indices': skill_indices,
        'internship_vectors': internship_vectors,
        'internship_descriptions': internship_descriptions,
        'interactions': interactions_df
    }

def _parse_skills(skills_str):
    """
    Parse skills string to list
    
    Args:
        skills_str (str): Skills string (comma-separated or JSON)
    
    Returns:
        list: List of skills
    """
    if pd.isna(skills_str):
        return []
    
    # Try parsing as JSON
    try:
        return json.loads(skills_str)
    except:
        # If not JSON, split by comma
        return [skill.strip() for skill in skills_str.split(',')]

def _create_sample_data(data_dir):
    """
    Create sample data for demo purposes
    
    Args:
        data_dir (str): Directory to save data
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample internships
    internships = [
        {
            'id': 1,
            'title': 'Data Science Intern',
            'company': 'TechCorp',
            'description': 'Looking for a motivated data science intern to work on machine learning projects. You will analyze data, build models, and contribute to production ML systems.',
            'required_skills': 'Python, Machine Learning, SQL, Data Analysis',
            'location': 'Paris',
            'duration': '6 months',
            'salary': '1000€/month'
        },
        {
            'id': 2,
            'title': 'Web Development Intern',
            'company': 'WebSolutions',
            'description': 'Join our web development team to create modern, responsive web applications using the latest frameworks and technologies.',
            'required_skills': 'JavaScript, HTML, CSS, React, Node.js',
            'location': 'Lyon',
            'duration': '4 months',
            'salary': '900€/month'
        },
        {
            'id': 3,
            'title': 'Business Analytics Intern',
            'company': 'DataCorp',
            'description': 'Help our business team analyze market trends, customer behavior, and business performance to drive strategic decisions.',
            'required_skills': 'Excel, SQL, Data Visualization, Business Intelligence',
            'location': 'Marseille',
            'duration': '3 months',
            'salary': '850€/month'
        },
        {
            'id': 4,
            'title': 'Machine Learning Research Intern',
            'company': 'AI Research Lab',
            'description': 'Work with our research team on cutting-edge machine learning algorithms, focusing on NLP and computer vision applications.',
            'required_skills': 'Python, PyTorch, TensorFlow, Deep Learning, NLP',
            'location': 'Paris',
            'duration': '6 months',
            'salary': '1200€/month'
        },
        {
            'id': 5,
            'title': 'Mobile App Development Intern',
            'company': 'AppFactory',
            'description': 'Develop native mobile applications for iOS and Android platforms, working on real-world products used by thousands of users.',
            'required_skills': 'Java, Kotlin, Swift, Mobile Development',
            'location': 'Nice',
            'duration': '5 months',
            'salary': '950€/month'
        },
        {
            'id': 6,
            'title': 'Data Engineering Intern',
            'company': 'BigData Inc',
            'description': 'Join our data engineering team to build and maintain data pipelines, ETL processes, and data warehousing solutions.',
            'required_skills': 'Python, SQL, Apache Spark, ETL, Data Pipelines',
            'location': 'Lille',
            'duration': '6 months',
            'salary': '1100€/month'
        },
        {
            'id': 7,
            'title': 'UX/UI Design Intern',
            'company': 'DesignStudio',
            'description': 'Create beautiful and intuitive user interfaces for web and mobile applications, focusing on user experience and accessibility.',
            'required_skills': 'UI Design, UX Research, Figma, Adobe XD, Prototyping',
            'location': 'Bordeaux',
            'duration': '4 months',
            'salary': '900€/month'
        },
        {
            'id': 8,
            'title': 'Cybersecurity Intern',
            'company': 'SecureTech',
            'description': 'Work on identifying and addressing security vulnerabilities in our systems, implementing security measures, and conducting security audits.',
            'required_skills': 'Network Security, Penetration Testing, Security Auditing, Linux',
            'location': 'Paris',
            'duration': '6 months',
            'salary': '1150€/month'
        },
        {
            'id': 9,
            'title': 'DevOps Intern',
            'company': 'CloudSystems',
            'description': 'Join our DevOps team to automate deployment pipelines, manage cloud infrastructure, and improve system reliability.',
            'required_skills': 'Linux, Docker, Kubernetes, AWS, CI/CD',
            'location': 'Toulouse',
            'duration': '5 months',
            'salary': '1050€/month'
        },
        {
            'id': 10,
            'title': 'Product Management Intern',
            'company': 'ProductLab',
            'description': 'Work closely with our product team to define product requirements, conduct market research, and coordinate with development teams.',
            'required_skills': 'Product Strategy, Market Research, Agile, Analytics',
            'location': 'Strasbourg',
            'duration': '4 months',
            'salary': '950€/month'
        }
    ]
    
    # Create sample skills
    skills = []
    skill_id = 1
    all_skills = set()
    
    for internship in internships:
        intern_skills = _parse_skills(internship['required_skills'])
        all_skills.update(intern_skills)
    
    for skill in sorted(all_skills):
        skills.append({
            'id': skill_id,
            'name': skill,
            'category': _categorize_skill(skill)
        })
        skill_id += 1
    
    # Save to CSV
    pd.DataFrame(internships).to_csv(os.path.join(data_dir, 'internships.csv'), index=False)
    pd.DataFrame(skills).to_csv(os.path.join(data_dir, 'skills.csv'), index=False)
    
    # Create empty interactions file
    pd.DataFrame(columns=['user_id', 'internship_id', 'rating', 'timestamp']).to_csv(
        os.path.join(data_dir, 'user_interactions.csv'), index=False
    )
    
    print(f"Sample data created in {data_dir}")

def _categorize_skill(skill):
    """
    Categorize a skill into a domain
    
    Args:
        skill (str): Skill name
    
    Returns:
        str: Skill category
    """
    programming_languages = ['Python', 'Java', 'JavaScript', 'Kotlin', 'Swift', 'C++', 'C#', 'Ruby']
    data_science = ['Machine Learning', 'Data Analysis', 'Deep Learning', 'NLP', 'TensorFlow', 'PyTorch', 'Statistics']
    web_dev = ['HTML', 'CSS', 'React', 'Node.js', 'Angular', 'Vue.js', 'Web Development', 'Frontend', 'Backend']
    database = ['SQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Oracle', 'NoSQL']
    devops = ['Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'CI/CD', 'DevOps', 'Linux']
    design = ['UI Design', 'UX Research', 'Figma', 'Adobe XD', 'Prototyping', 'Wireframing']
    business = ['Market Research', 'Business Intelligence', 'Product Strategy', 'Analytics', 'Excel']
    
    skill_lower = skill.lower()
    
    if any(lang.lower() in skill_lower for lang in programming_languages):
        return 'Programming'
    elif any(ds.lower() in skill_lower for ds in data_science):
        return 'Data Science'
    elif any(web.lower() in skill_lower for web in web_dev):
        return 'Web Development'
    elif any(db.lower() in skill_lower for db in database):
        return 'Database'
    elif any(ops.lower() in skill_lower for ops in devops):
        return 'DevOps'
    elif any(des.lower() in skill_lower for des in design):
        return 'Design'
    elif any(biz.lower() in skill_lower for biz in business):
        return 'Business'
    else:
        return 'Other'
