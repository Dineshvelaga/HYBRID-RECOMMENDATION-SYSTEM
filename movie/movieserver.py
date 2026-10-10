from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from surprise import Reader, Dataset, SVD, accuracy, Prediction
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.simplefilter('ignore')

app = Flask(__name__)

def process_genres(genre_str):
    """
    Helper function to process genres from string to list of genre names
    """
    try:
        if isinstance(genre_str, str):
            genre_list = literal_eval(genre_str)
            if isinstance(genre_list, list):
                return [g['name'] for g in genre_list if isinstance(g, dict) and 'name' in g]
        return []
    except:
        return []

# Define HybridModel class
class HybridModel:
    def __init__(self, svd_model, reconstructed_matrix, user_item_matrix, weight_svd=0.8, weight_tsvd=0.2):
        self.svd_model = svd_model
        self.reconstructed_matrix = reconstructed_matrix
        self.user_item_matrix = user_item_matrix
        self.weight_svd = weight_svd
        self.weight_tsvd = weight_tsvd
        # Fallback global mean rating in case user/item not found in the user_item_matrix
        self.global_mean = user_item_matrix.values.mean()
    
    def predict(self, uid, iid, r_ui=None):
        # Get SVD prediction
        svd_pred = self.svd_model.predict(uid, iid, r_ui)
        # Get TSVD prediction from the reconstructed matrix
        if uid in self.user_item_matrix.index and iid in self.user_item_matrix.columns:
            user_index = self.user_item_matrix.index.get_loc(uid)
            item_index = self.user_item_matrix.columns.get_loc(iid)
            tsvd_pred = self.reconstructed_matrix[user_index, item_index]
        else:
            tsvd_pred = self.global_mean
        # Combine the predictions using the specified weights
        hybrid_est = self.weight_svd * svd_pred.est + self.weight_tsvd * tsvd_pred
        # Return a Prediction object with the hybrid estimate
        return Prediction(uid, iid, r_ui, hybrid_est, None)

# Check for required files
base_path = 'datasets/'
required_files = ['movie_meta.csv', 'ratings_small.csv', 'links_small.csv']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(base_path, f))]

if missing_files:
    raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

# Load and preprocess data
try:
    print("Loading and preprocessing data...")
    md = pd.read_csv('datasets/movie_meta.csv')
    df_ratings = pd.read_csv('datasets/ratings_small.csv')
    links_small = pd.read_csv('datasets/links_small.csv')

    # Preprocess movie metadata
    print("Preprocessing movie metadata...")
    md['genres'] = md['genres'].fillna('[]')
    md['genres'] = md['genres'].apply(process_genres)
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').dt.year

    # Drop problematic rows
    drop_indices = [19730, 29503, 35587]
    md = md.drop(drop_indices, errors='ignore')
    md = md.dropna(subset=['id'])
    md['id'] = md['id'].astype(int)

    # Prepare for content-based recommendations
    print("Preparing content-based recommendations...")
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    smd = md[md['id'].isin(links_small)]
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = (smd['overview'] + " " + smd['tagline']).fillna('')

    # TF-IDF Vectorizer for content-based recommender
    print("Training TF-IDF model...")
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Reset index and create reverse mapping from title to index
    smd = smd.reset_index(drop=True)
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])

    # Prepare for collaborative filtering
    print("Preparing collaborative filtering models...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Train SVD model
    print("Training SVD model...")
    svd_model = SVD()
    svd_model.fit(trainset)

    # Build user-item matrix for TSVD
    print("Building user-item matrix...")
    user_item_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(user_item_matrix)

    # Train TSVD model
    print("Training TSVD model...")
    tsvd_model = TruncatedSVD(n_components=250)
    reduced_matrix = tsvd_model.fit_transform(scaled_matrix)
    reconstructed_matrix = np.dot(reduced_matrix, tsvd_model.components_)
    reconstructed_matrix = scaler.inverse_transform(reconstructed_matrix)

    # Create hybrid model
    print("Creating hybrid model...")
    model = HybridModel(svd_model, reconstructed_matrix, user_item_matrix)
    
    print("All models trained successfully!")
    
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise

# Add statistics
total_movies = len(md)
total_users = len(df_ratings['userId'].unique())
total_ratings = len(df_ratings)
avg_rating = df_ratings['rating'].mean()

def get_recommendations(userId):
    """
    Provides movie recommendations for a user by combining content-based and collaborative filtering.
    If the user is new (cold start), it provides popular movie recommendations.
    """
    if userId not in df_ratings['userId'].unique():
        # New user: Recommend top-rated movies (popularity-based)
        popular_movies = md[['title', 'vote_count', 'vote_average']].dropna()
        popular_movies = popular_movies[popular_movies['vote_count'] >= popular_movies['vote_count'].quantile(0.60)]
        popular_movies['score'] = popular_movies.apply(
            lambda x: (x['vote_count'] / (x['vote_count'] + 100) * x['vote_average']) + 
                      (100 / (100 + x['vote_count']) * popular_movies['vote_average'].mean()), axis=1)
        popular_movies = popular_movies.sort_values('score', ascending=False)
        return popular_movies[['title']].head(10)
    else:
        # Existing user: Hybrid recommendation
        user_ratings = df_ratings[df_ratings['userId'] == userId]
        user_ratings = user_ratings.merge(md[['id', 'title']], left_on='movieId', right_on='id')
        user_ratings = user_ratings.sort_values(by='rating', ascending=False).head(5)
        
        recommendations = set()
        for _, row in user_ratings.iterrows():
            similar_movies = get_content_based_recommendations(row['title'])
            if similar_movies:
                recommendations.update(similar_movies)
        
        recommendations = list(recommendations)
        predicted_ratings = []
        for title in recommendations:
            movie_id = md[md['title'] == title]['id'].values[0]
            pred_rating = model.predict(userId, movie_id).est
            predicted_ratings.append((title, pred_rating))
        
        # Create a DataFrame to display titles and predicted ratings
        recommendations_df = pd.DataFrame(predicted_ratings, columns=['title', 'predicted_rating'])
        recommendations_df = recommendations_df.sort_values(by='predicted_rating', ascending=False).head(10)
        
        return recommendations_df

@app.route('/')
def index():
    return render_template('index.html', 
                         total_movies=total_movies,
                         total_users=total_users,
                         total_ratings=total_ratings,
                         avg_rating=round(avg_rating, 2))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/search_movies')
def search_movies():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify({'movies': []})
    
    matching_movies = md[md['title'].str.lower().str.contains(query, na=False)]
    movies = matching_movies[['title', 'genres', 'release_date']].head(10)
    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
    
    return jsonify({
        'movies': movies.to_dict('records')
    })

def get_content_based_recommendations(title, n_recs=10):
    """
    Given a movie title, return the top n_recs most similar movies based on cosine similarity.
    """
    if title not in indices:
        return None
    # If multiple indices exist for the title, take the first one
    idx = indices[title]
    if isinstance(idx, (pd.Series, np.ndarray)):
        idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
        
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recs+1]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices].tolist()

def analyze_and_recommend(user_ids, recommend_new=True):
    """
    Provides movie recommendations based on the SVD model.
    """
    # Check for valid user IDs
    valid_users = [uid for uid in user_ids if uid in df_ratings['userId'].unique()]
    missing_users = [uid for uid in user_ids if uid not in df_ratings['userId'].unique()]
    
    if missing_users:
        print(f"Warning: The following user IDs are not found in the ratings dataset: {missing_users}")
    if not valid_users:
        print("No valid user IDs provided. Recommending popular movies.")
        # Fallback: Popularity-based recommendation
        popular_movies = md[['title', 'vote_count', 'vote_average']].dropna()
        popular_movies['vote_count'] = pd.to_numeric(popular_movies['vote_count'], errors='coerce')
        popular_movies = popular_movies.dropna(subset=['vote_count'])
        threshold_vote = popular_movies['vote_count'].quantile(0.60)
        popular_movies = popular_movies[popular_movies['vote_count'] >= threshold_vote]
        popular_movies['predicted_rating'] = popular_movies.apply(
            lambda x: (x['vote_count']/(x['vote_count']+100) * x['vote_average']) + 
                      (100/(100+x['vote_count']) * popular_movies['vote_average'].mean()), axis=1)
        popular_movies = popular_movies.sort_values('predicted_rating', ascending=False)
        return popular_movies[['title', 'predicted_rating']].head(10)
    
    # Use only valid user IDs for further processing
    user_ids = valid_users
    # Combined ratings for these users
    combined_ratings = df_ratings[df_ratings['userId'].isin(user_ids)]
    seen_movies = combined_ratings['movieId'].unique()
    
    if recommend_new:
        # For new movie recommendations: Candidate movies are those not seen by any of the users
        candidate_movies = md[~md['id'].isin(seen_movies)]
    else:
        # For seen movies: Candidate movies are those that at least one user has watched
        candidate_movies = md[md['id'].isin(seen_movies)]
    
    # Compute the average predicted rating for each candidate movie across all valid users
    def avg_pred(movie_id):
        # For each user, get the predicted rating for the movie
        preds = [model.predict(user, movie_id).est for user in user_ids]
        return np.mean(preds)
    
    candidate_movies = candidate_movies.copy()
    candidate_movies['predicted_rating'] = candidate_movies['id'].apply(avg_pred)
    candidate_movies = candidate_movies.sort_values('predicted_rating', ascending=False)
    
    # Return top 10 recommendations along with predicted ratings
    return candidate_movies[['title', 'predicted_rating']].head(10)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        recommendation_type = data.get('type')
        
        if recommendation_type == 'single_user':
            user_id = data.get('userId')
            if not user_id:
                return jsonify({'error': 'userId is required for single user recommendations'}), 400
            recommendations = get_recommendations(user_id)
            if isinstance(recommendations, pd.DataFrame):
                recommendations = recommendations.merge(md[['title', 'genres', 'release_date']], on='title')
                recommendations['genres'] = recommendations['genres'].apply(lambda x: process_genres(x))
                recommendations['year'] = pd.to_datetime(recommendations['release_date'], errors='coerce').dt.year
                return jsonify({'recommendations': recommendations.to_dict('records')})
            else:
                return jsonify({'error': 'No recommendations found'}), 404
            
        elif recommendation_type == 'multi_user':
            user_ids = data.get('userIds')
            recommend_new = data.get('recommendNew', True)
            if not user_ids:
                return jsonify({'error': 'userIds is required for multi user recommendations'}), 400
            recommendations = analyze_and_recommend(user_ids, recommend_new)
            if isinstance(recommendations, pd.DataFrame):
                recommendations = recommendations.merge(md[['title', 'genres', 'release_date']], on='title')
                recommendations['genres'] = recommendations['genres'].apply(lambda x: process_genres(x))
                recommendations['year'] = pd.to_datetime(recommendations['release_date'], errors='coerce').dt.year
                return jsonify({'recommendations': recommendations.to_dict('records')})
            else:
                return jsonify({'error': 'No recommendations found'}), 404
            
        elif recommendation_type == 'content_based':
            movie_title = data.get('movieTitle')
            n_recs = data.get('nRecs', 10)
            if not movie_title:
                return jsonify({'error': 'movieTitle is required for content-based recommendations'}), 400
            recommendations = get_content_based_recommendations(movie_title, n_recs)
            if recommendations:
                recommendations_df = pd.DataFrame(recommendations, columns=['title'])
                recommendations_df = recommendations_df.merge(md[['title', 'genres', 'release_date']], on='title')
                recommendations_df['genres'] = recommendations_df['genres'].apply(lambda x: process_genres(x))
                recommendations_df['year'] = pd.to_datetime(recommendations_df['release_date'], errors='coerce').dt.year
                return jsonify({'recommendations': recommendations_df.to_dict('records')})
            else:
                return jsonify({'error': 'No recommendations found'}), 404
            
        else:
            return jsonify({'error': 'Invalid recommendation type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
