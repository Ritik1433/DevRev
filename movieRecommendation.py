import pandas as pd
import numpy as np

# Load the MovieLens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge the datasets
movie_data = pd.merge(ratings, movies, on='movieId')

# Group the data by user ID and movie title
user_movie_data = movie_data.groupby(['userId', 'title']).agg({'rating': ['mean']})

# Flatten the multi-level index
user_movie_data.columns = user_movie_data.columns.droplevel(level=1)
user_movie_data.reset_index(inplace=True)

# Create a pivot table with user IDs as rows and movie titles as columns
user_movie_matrix = user_movie_data.pivot_table(index='userId', columns='title', values='rating')

# Remove movies with fewer than 10 ratings
user_movie_matrix = user_movie_matrix.dropna(thresh=10, axis=1)

# Fill missing values with 0
user_movie_matrix.fillna(0, inplace=True)

# Calculate cosine similarity between movie ratings
from sklearn.metrics.pairwise import cosine_similarity
movie_similarity_matrix = cosine_similarity(user_movie_matrix.T)

# Convert the similarity matrix into a dataframe
movie_similarity_df = pd.DataFrame(movie_similarity_matrix, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Define a function to recommend movies
def recommend_movies(movie_title):
    similarity_series = movie_similarity_df[movie_title].sort_values(ascending=False)
    return similarity_series.iloc[1:11].index.tolist()

# Example: Recommend movies similar to 'The Dark Knight'
recommendations = recommend_movies('The Dark Knight (2008)')
print(recommendations)
