from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import pickle
from functools import lru_cache
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
DATA_PATH = BASE_DIR / 'datasets' / 'ex.csv'
MODEL4_DATA_PATH = MODELS_DIR / 'df.pkl'
MODEL4_SIMILARITY_PATH = MODELS_DIR / 'similarity.pkl'
FALLBACK_COVER_URL = 'https://i.postimg.cc/0QNxYz4V/social.png'

app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Initialize Spotify client
if CLIENT_ID and CLIENT_SECRET:
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(
        client_credentials_manager=client_credentials_manager,
        requests_timeout=2,
        retries=0,
        status_retries=0
    )
else:
    sp = None

print("Loading Model 4 data...")
# Load Model 4 data
try:
    with MODEL4_DATA_PATH.open('rb') as model_file:
        music_df = pickle.load(model_file)
    with MODEL4_SIMILARITY_PATH.open('rb') as similarity_file:
        similarity_matrix = pickle.load(similarity_file)
    print(f"Model 4 data loaded successfully:")
    print(f"- Number of songs: {len(music_df)}")
    print(f"- Columns available: {music_df.columns.tolist()}")
    print(f"- First few songs: {music_df['song'].head().tolist()}")
except Exception as e:
    print(f"Error loading Model 4 data: {e}")
    music_df = None
    similarity_matrix = None

# Load data
df = pd.read_csv(DATA_PATH)

# Fill NaN values with empty strings
df['Song-Name'] = df['Song-Name'].fillna('')
df['Singer/Artists'] = df['Singer/Artists'].fillna('')
df['Album/Movie'] = df['Album/Movie'].fillna('')
df['Genre'] = df['Genre'].fillna('')

# Create tags by combining relevant columns
df['tags'] = df['Song-Name'] + ' ' + df['Singer/Artists'] + ' ' + df['Album/Movie'] + ' ' + df['Genre']

# Create new dataframe with required columns
new_df = df[['Song-Name', 'Singer/Artists', 'Genre', 'Album/Movie', 'tags']].copy()
new_df.rename(columns={'Song-Name':'title'}, inplace=True)

@lru_cache(maxsize=1024)
def get_song_album_cover_url(song_name, artist_name):
    if sp is None:
        return FALLBACK_COVER_URL

    try:
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track")

        if results and results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            album_cover_url = track["album"]["images"][0]["url"]
            return album_cover_url
        else:
            return FALLBACK_COVER_URL
    except:
        return FALLBACK_COVER_URL

# Load saved models
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_paths = {
    'count_vectorizer': MODELS_DIR / 'count_vectorizer.joblib',
    'tfidf_vectorizer': MODELS_DIR / 'tfidf_vectorizer.joblib',
    'similarity1': MODELS_DIR / 'similarity1.joblib',
    'similarity2': MODELS_DIR / 'similarity2.joblib',
    'similarity3': MODELS_DIR / 'similarity3.joblib',
}

if not all(path.exists() for path in model_paths.values()):
    # Train and save models if they don't exist
    count_vectorizer = CountVectorizer(max_features=2000)
    tfidf_vectorizer = TfidfVectorizer()
    
    count_matrix = count_vectorizer.fit_transform(new_df['tags'])
    tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['tags'])
    
    similarity1 = cosine_similarity(count_matrix)
    similarity2 = cosine_similarity(tfidf_matrix)
    similarity3 = (similarity1 + similarity2) / 2
    
    joblib.dump(count_vectorizer, model_paths['count_vectorizer'])
    joblib.dump(tfidf_vectorizer, model_paths['tfidf_vectorizer'])
    joblib.dump(similarity1, model_paths['similarity1'])
    joblib.dump(similarity2, model_paths['similarity2'])
    joblib.dump(similarity3, model_paths['similarity3'])
else:
    # Load existing models
    count_vectorizer = joblib.load(model_paths['count_vectorizer'])
    tfidf_vectorizer = joblib.load(model_paths['tfidf_vectorizer'])
    similarity1 = joblib.load(model_paths['similarity1'])
    similarity2 = joblib.load(model_paths['similarity2'])
    similarity3 = joblib.load(model_paths['similarity3'])

def recommend(music, model=3):
    """Recommends top 5 similar songs to the provided music title."""
    try:
        # Select the appropriate similarity matrix
        if model == 1:
            sim_matrix = similarity1
            df = new_df
            title_col = 'title'
        elif model == 2:
            sim_matrix = similarity2
            df = new_df
            title_col = 'title'
        elif model == 3:
            sim_matrix = similarity3
            df = new_df
            title_col = 'title'
        elif model == 4 and music_df is not None and similarity_matrix is not None:
            # Model 4 recommendation logic
            try:
                index = music_df[music_df['song'] == music].index[0]
                distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
                recommendations = []
                for i in distances[1:6]:
                    artist = music_df.iloc[i[0]].artist
                    song_name = music_df.iloc[i[0]].song
                    cover_url = get_song_album_cover_url(song_name, artist)
                    recommendations.append({
                        'title': song_name,
                        'artist': artist,
                        'genre': '',
                        'album': '',
                        'score': float(i[1]),
                        'cover_url': cover_url
                    })
                return recommendations
            except Exception as e:
                print(f"Error in Model 4 recommendation: {e}")
                return None
        else:
            return None

        # Find the index for the given song title
        try:
            music_index = df[df[title_col] == music].index[0]
        except IndexError:
            print(f"Song '{music}' not found in dataset")
            return None

        # Get similarity scores and sort them
        distances = sim_matrix[music_index]
        music_lists = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommendations = []
        for idx, score in music_lists:
            cover_url = get_song_album_cover_url(
                df.iloc[idx]['title'],
                df.iloc[idx]['Singer/Artists']
            )
            recommendations.append({
                'title': df.iloc[idx]['title'],
                'artist': df.iloc[idx]['Singer/Artists'],
                'genre': df.iloc[idx]['Genre'],
                'album': df.iloc[idx]['Album/Movie'],
                'score': float(score),
                'cover_url': cover_url
            })

        return recommendations

    except Exception as e:
        print(f"Error in recommendation function: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_songs():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'error': 'Please provide a search query'}), 400
    
    # Search in all relevant columns
    results = new_df[
        (new_df['title'].str.lower().str.contains(query)) |
        (new_df['Singer/Artists'].str.lower().str.contains(query)) |
        (new_df['Genre'].str.lower().str.contains(query)) |
        (new_df['Album/Movie'].str.lower().str.contains(query))
    ]
    
    return jsonify({
        'query': query,
        'results': results[['title', 'Singer/Artists', 'Genre', 'Album/Movie']].to_dict('records')
    })

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    song = request.args.get('song')
    model = request.args.get('model', default=3, type=int)
    
    if not song:
        return jsonify({'error': 'Please provide a song title'}), 400
    
    recommendations = recommend(song, model)
    if recommendations is None:
        if model == 4:
            return jsonify({'error': f'Song "{song}" not found in Spotify dataset'}), 404
        else:
            return jsonify({'error': f'Song "{song}" not found in dataset'}), 404
    
    return jsonify({
        'song': song,
        'model': model,
        'recommendations': recommendations
    })

@app.route('/songs', methods=['GET'])
def get_songs():
    return jsonify({
        'songs': new_df[['title', 'Singer/Artists', 'Genre', 'Album/Movie']].to_dict('records')
    })

@app.route('/model4-songs', methods=['GET'])
def get_model4_songs():
    try:
        if music_df is not None:
            print("Model 4 songs count:", len(music_df))
            songs = music_df[['song', 'artist']].to_dict('records')
            print("First few songs:", songs[:5])  # Debug print
            return jsonify({
                'songs': songs
            })
        else:
            print("Model 4 data (music_df) is None")
            return jsonify({'error': 'Model 4 data not available'}), 404
    except Exception as e:
        print(f"Error in get_model4_songs: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5001) 