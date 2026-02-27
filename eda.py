import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

print("Ratings Shape:", ratings.shape)
print("Movies Shape:", movies.shape)

print("Unique Users:", ratings['userId'].nunique())
print("Unique Movies:", ratings['movieId'].nunique())

# Rating Distribution
plt.figure()
sns.countplot(x='rating', data=ratings)
plt.title("Rating Distribution")
plt.show()

# Sparsity Calculation
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
num_ratings = len(ratings)

sparsity = 1 - (num_ratings / (num_users * num_movies))
print("Dataset Sparsity:", sparsity)