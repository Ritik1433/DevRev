# DevRev
solution for problem statement 1
Explanation:

Load the MovieLens dataset using Pandas' read_csv function.

Merge the ratings and movies datasets using the merge function.

Group the data by user ID and movie title, and calculate the mean rating for each movie.

Flatten the multi-level index and create a pivot table with user IDs as rows and movie titles as columns.

Remove movies with fewer than 10 ratings and fill missing values with 0.

Calculate the cosine similarity between movie ratings using the cosine_similarity function from Scikit-learn.

Convert the similarity matrix into a dataframe and define a function to recommend movies based on the similarity scores.

Use the recommend_movies function to recommend movies similar to 'The Dark Knight'.