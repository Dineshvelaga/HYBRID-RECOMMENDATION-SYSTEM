import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise import accuracy
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

#new
# Load datasets
md = pd.read_csv('movie_meta.csv')
df_ratings = pd.read_csv('ratings_small.csv')
links_small = pd.read_csv('links_small.csv')
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

ratings = pd.read_csv('ratings_small.csv')
ratings['rating'].plot.hist()
#Set axis labels and title
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Histogram of overall ratings')
plt.show()

import pandas as pd
import numpy as np
from ast import literal_eval

# Load the dataset (assuming 'md' is already a DataFrame)
md['genres'] = md['genres'].fillna('[]')

# Apply literal_eval only if the value is a string
md['genres'] = md['genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

# Extract genre names
md['genres'] = md['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# Process release_date and extract year
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').dt.year

# Drop problematic rows; ignore errors if indices are not found
drop_indices = [19730, 29503, 35587]
md = md.drop(drop_indices, errors='ignore')

# Drop rows where 'id' is NaN and convert 'id' column to integer
md = md.dropna(subset=['id'])
md['id'] = md['id'].astype(int)

# Check the first few rows to verify
md.head()

# Use links_small to filter movies in md
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
smd = md[md['id'].isin(links_small)]

# Fill missing tagline and create description from overview and tagline
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = (smd['overview'] + " " + smd['tagline']).fillna('')

# TF-IDF Vectorizer for content-based recommender
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reset index and create reverse mapping from title to index
smd = smd.reset_index(drop=True)
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

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

get_content_based_recommendations('The Godfather: Part II')

reader = Reader(rating_scale=(0.5, 5.0))
#max rating =5 and min rating=0.5
data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)
data

import time  # Import time module
start_time = time.time()  # Start time

# Your Python code here

end_time = time.time()  # End time
print(f"Execution time: {end_time - start_time} seconds")
from surprise.model_selection import train_test_split
from surprise import accuracy, Dataset
from surprise import SVD

# Assuming 'data' is already defined
# Split off 30% of the data for testing
_, testset = train_test_split(data, test_size=0.30)

# Train the model on the full dataset%%time
from surprise.model_selection import train_test_split
from surprise import accuracy, Dataset, SVD
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from surprise import Prediction

# Assuming 'data' is already defined (a Surprise Dataset) 
# and 'df' is a pandas DataFrame with columns: 'userId', 'movieId', 'rating'

# Split off 30% of the data for testing
_, testset = train_test_split(data, test_size=0.30)

# Train the SVD model on the full dataset
trainset = data.build_full_trainset()
svd_model = SVD()
svd_model.fit(trainset)

# Build user-item matrix for TSVD from the DataFrame
user_item_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(user_item_matrix)

# Train the TSVD model
tsvd_model = TruncatedSVD(n_components=250)
reduced_matrix = tsvd_model.fit_transform(scaled_matrix)
reconstructed_matrix = np.dot(reduced_matrix, tsvd_model.components_)
reconstructed_matrix = scaler.inverse_transform(reconstructed_matrix)

# Define a HybridModel that combines SVD and TSVD predictions
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

# Create the hybrid model using 0.8 weight for SVD and 0.2 for TSVD
model = HybridModel(svd_model, reconstructed_matrix, user_item_matrix, weight_svd=0.8, weight_tsvd=0.2)

# Generate predictions on the test set using the hybrid model
hybrid_predictions = [model.predict(uid, iid, true_r) for (uid, iid, true_r) in testset]

# Evaluate the hybrid model
accuracy.rmse(hybrid_predictions, verbose=True)
accuracy.mae(hybrid_predictions, verbose=True)

def get_content_based_recommendation_for_user(userId):
    # Check if user id exist
    # if not exist show some of populars

    if userId not in df_ratings_merge['userId'].unique():
      #highest_ranked_movies = df_ratings_merge.nlargest(15, 'rating')
      # Sort the dataframe by the "rating" column in descending order and select the top 15 rows
      highest_ranked_movies = df_ratings_merge.sort_values(by='rating', ascending=False).head(15)
      # Ensure that the "title" values are unique
      highest_ranked_movies = highest_ranked_movies.drop_duplicates(subset='title')
      print("new user")
      print(highest_ranked_movies)
    else:
      # Get highest ranking score of this user
      highest_ranked_movies = df_ratings_merge[df_ratings_merge['userId'] == userId]
      highest_ranked_movies.sort_values(by=['rating'], ascending=False)
      print(highest_ranked_movies)
      
    similar_movies = set()
    for i in range(len(highest_ranked_movies)):
        current_similar_movies = get_content_based_recommendations(highest_ranked_movies.iloc[i].title)
        if current_similar_movies == None:
            continue
        for movie_title in current_similar_movies:
            similar_movies.add(movie_title)
    
    result_df = pd.DataFrame(similar_movies, columns=['title'])
    
    without_duplicate = df_ratings_merge_with_different_title.drop_duplicates(subset='title', keep='first', inplace = False)    
    result_df = result_df.merge(without_duplicate, on = 'title')
    
    return result_df

def get_recommendations(userId):
    """
    Provides movie recommendations for a user by combining content-based and collaborative filtering.
    If the user is new (cold start), it provides popular movie recommendations.
    """
    ratings = pd.read_csv('ratings_small.csv')
    if userId not in ratings['userId'].unique():
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
        user_ratings = ratings[ratings['userId'] == userId]
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
    
# For an existing user
print("Recommendations for User 150:")
print(get_recommendations(150))

# For a new user
print("\nRecommendations for a new user (User 10000):")
print(get_recommendations(10000))

#multi user group recommendations
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

# --------------------------
# Load datasets
# --------------------------
md = pd.read_csv('movie_meta.csv', low_memory=False)
ratings = pd.read_csv('ratings_small.csv')

# Ensure that the 'id' column in md is numeric for later merging/comparison
md['id'] = pd.to_numeric(md['id'], errors='coerce')
md = md.dropna(subset=['id'])
md['id'] = md['id'].astype(int)

# --------------------------
# Preprocess and prepare the data for Surprise
# --------------------------
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Optional: Train-test split (here used only for evaluation)
X = ratings.drop(columns='rating')
y = ratings['rating']
# Note: The following split is not used by SVD, which is trained on the full trainset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Train SVD model
# --------------------------
svd = SVD()
svd.fit(trainset)

# Evaluate the SVD model using the built-in Surprise testset
testset = trainset.build_testset()
predictions = svd.test(testset)

# Calculate RMSE and MAE
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

# Prepare for classification metrics (binarize using threshold)
y_true = [pred.r_ui for pred in predictions]
y_pred = [pred.est for pred in predictions]
threshold = 3.5
y_pred_binary = [1 if x >= threshold else 0 for x in y_pred]
y_true_binary = [1 if x >= threshold else 0 for x in y_true]

precision = precision_score(y_true_binary, y_pred_binary, average='micro')
recall = recall_score(y_true_binary, y_pred_binary, average='micro')
accuracy_val = accuracy_score(y_true_binary, y_pred_binary)

results = {
    'RMSE': rmse,
    'MAE': mae,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy_val
}

print(f"SVD Results: {results}")

# --------------------------
# Recommendation Function
# --------------------------
def analyze_and_recommend(user_ids, recommend_new=True):
    """
    Provides movie recommendations based on the SVD model.
    
    - For new movies (recommend_new=True): recommends movies that none of the given users have seen.
    - For seen movies (recommend_new=False): recommends movies that at least one of the users has watched.
    
    The function checks that all provided user IDs exist in the ratings data. If some are missing,
    a warning is printed. If no valid user IDs remain, the function falls back to popularity-based recommendations.
    """
    # Check for valid user IDs
    valid_users = [uid for uid in user_ids if uid in ratings['userId'].unique()]
    missing_users = [uid for uid in user_ids if uid not in ratings['userId'].unique()]
    
    if missing_users:
        print(f"Warning: The following user IDs are not found in the ratings dataset: {missing_users}")
    if not valid_users:
        print("No valid user IDs provided. Recommending popular movies.")
        # Fallback: Popularity-based recommendation
        popular_movies = md[['title', 'vote_count', 'vote_average']].dropna()
        # Convert vote_count to numeric if necessary
        popular_movies['vote_count'] = pd.to_numeric(popular_movies['vote_count'], errors='coerce')
        popular_movies = popular_movies.dropna(subset=['vote_count'])
        # Use a threshold (e.g., 60th percentile)
        threshold_vote = popular_movies['vote_count'].quantile(0.60)
        popular_movies = popular_movies[popular_movies['vote_count'] >= threshold_vote]
        # Compute a weighted score as an example
        popular_movies['predicted_rating'] = popular_movies.apply(
            lambda x: (x['vote_count']/(x['vote_count']+100) * x['vote_average']) + 
                      (100/(100+x['vote_count']) * popular_movies['vote_average'].mean()), axis=1)
        popular_movies = popular_movies.sort_values('predicted_rating', ascending=False)
        return popular_movies[['title', 'predicted_rating']].head(10)
    
    # Use only valid user IDs for further processing
    user_ids = valid_users
    # Combined ratings for these users
    combined_ratings = ratings[ratings['userId'].isin(user_ids)]
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
        preds = [svd.predict(user, movie_id).est for user in user_ids]
        return np.mean(preds)
    
    candidate_movies = candidate_movies.copy()
    candidate_movies['predicted_rating'] = candidate_movies['id'].apply(avg_pred)
    candidate_movies = candidate_movies.sort_values('predicted_rating', ascending=False)
    
    # Return top 10 recommendations along with predicted ratings
    return candidate_movies[['title', 'predicted_rating']].head(10)

# --------------------------
# Example Usage
# --------------------------


user_ids = [1, 2,51]  # Modify as needed
print("SVD Recommendations for new movies (not seen by any user):")
print(analyze_and_recommend(user_ids, recommend_new=True))

print("\nSVD Recommendations for already watched movies (seen by at least one user):")
print(analyze_and_recommend(user_ids, recommend_new=False))

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#import seaborn as sns

# Load datasets
md = pd.read_csv('movie_meta.csv', low_memory=False)
ratings = pd.read_csv('ratings_small.csv')

# Preprocess and prepare the data
md['id'] = pd.to_numeric(md['id'], errors='coerce')
md = md.dropna(subset=['id'])
md['id'] = md['id'].astype('int')

reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train-test split for sklearn models
train_data = ratings[['userId', 'movieId', 'rating']]
X = train_data.drop(columns='rating')
y = train_data['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train different models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVD': SVD(),
    'KNN': KNNBasic()
}

results = {}

for name, model in models.items():
    if name == 'SVD' or name == 'KNN':
        # Fit Surprise models
        model.fit(trainset)
        testset = trainset.build_testset()
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        y_true = [pred.r_ui for pred in predictions]
        y_pred = [pred.est for pred in predictions]
        
        # Binarize predictions for classification metrics
        y_pred_binary = [1 if x >= 3.5 else 0 for x in y_pred]
        y_true_binary = [1 if x >= 3.5 else 0 for x in y_true]
        
        precision = precision_score(y_true_binary, y_pred_binary, average='micro')
        recall = recall_score(y_true_binary, y_pred_binary, average='micro')
        accuracy_val = accuracy_score(y_true_binary, y_pred_binary)
        
    else:
        # Fit sklearn models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Binarize predictions for classification metrics
        y_pred_binary = [1 if x >= 3.5 else 0 for x in y_pred]
        y_test_binary = [1 if x >= 3.5 else 0 for x in y_test]
        
        precision = precision_score(y_test_binary, y_pred_binary, average='micro')
        recall = recall_score(y_test_binary, y_pred_binary, average='micro')
        accuracy_val = accuracy_score(y_test_binary, y_pred_binary)
    
    results[name] = {'RMSE': rmse, 'MAE': mae, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy_val}

# Convert results to DataFrame for plotting
results_df = pd.DataFrame(results).T

# Plotting
plt.figure(figsize=(14, 8))

# Plot RMSE
plt.subplot(2, 2, 1)
#sns.barplot(x=results_df.index, y='RMSE', data=results_df, palette='viridis')
plt.title('RMSE of Models')
plt.ylabel('RMSE')
plt.xticks(rotation=45)

# Plot MAE
plt.subplot(2, 2, 2)
#sns.barplot(x=results_df.index, y='MAE', data=results_df, palette='viridis')
plt.title('MAE of Models')
plt.ylabel('MAE')
plt.xticks(rotation=45)

# Plot Precision
plt.subplot(2, 2, 3)
#sns.barplot(x=results_df.index, y='Precision', data=results_df, palette='viridis')
plt.title('Precision of Models')
plt.ylabel('Precision')
plt.xticks(rotation=45)

# Plot Recall
plt.subplot(2, 2, 4)
#sns.barplot(x=results_df.index, y='Recall', data=results_df, palette='viridis')
plt.title('Recall of Models')
plt.ylabel('Recall')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

