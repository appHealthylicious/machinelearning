import pickle
import json
import pandas as pd
import numpy as np
import heapq
from surprise import AlgoBase, PredictionImpossible
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets (make sure to update the paths accordingly)
recipes_df = pd.read_csv('C:/Users/arsen/Healthylicious/data/cleaned/csv/recipes_dataset.csv')
ratings_df = pd.read_csv('C:/Users/arsen/Healthylicious/data/cleaned/csv/ratings_dataset.csv')

# Convert relevant columns to integers
recipes_df['recipeId'] = recipes_df['recipeId'].astype(int)
ratings_df['recipeId'] = ratings_df['recipeId'].astype(int)
ratings_df['userId'] = ratings_df['userId'].astype(int)

# Initialize TF-IDF Vectorizer and compute TF-IDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','))
tfidf_matrix = vectorizer.fit_transform(recipes_df['Ingredients'])

class ContentKNNAlgorithm(AlgoBase):
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.recipes = {row['recipeId']: row for _, row in recipes_df.iterrows()}
        self.recipes_index = {row['recipeId']: i for i, row in recipes_df.iterrows()}
        
        print("Computing content-based similarity matrix...")
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        for thisRating in range(self.trainset.n_items):
            if (thisRating % 100 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating + 1, self.trainset.n_items):
                thisRecipeID = int(self.trainset.to_raw_iid(thisRating))
                otherRecipeID = int(self.trainset.to_raw_iid(otherRating))
                similarity = self.computeContentSimilarity(thisRecipeID, otherRecipeID)
                self.similarities[thisRating, otherRating] = similarity
                self.similarities[otherRating, thisRating] = similarity
        print("...done.")
        return self
    
    def computeContentSimilarity(self, recipe1_id, recipe2_id):
        recipe1 = self.recipes[recipe1_id]
        recipe2 = self.recipes[recipe2_id]
        recipe1_index = self.recipes_index[recipe1_id]
        recipe2_index = self.recipes_index[recipe2_id]
        ingredient_similarity = cosine_similarity(tfidf_matrix[recipe1_index], tfidf_matrix[recipe2_index]).flatten()[0]
        category_similarity = 1 if recipe1['Category'] == recipe2['Category'] else 0
        time_diff = abs(recipe1['Total Time'] - recipe2['Total Time'])
        time_similarity = np.exp(-time_diff / 10.0)
        combined_similarity = 0.2 * ingredient_similarity + 0.6 * category_similarity + 0.2 * time_similarity
        return combined_similarity
    
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        neighbors = []
        for rating in self.trainset.ur[u]:
            content_similarity = self.similarities[i, rating[0]]
            neighbors.append((content_similarity, rating[1]))
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        if not k_neighbors:
            raise PredictionImpossible('No neighbors')
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if simScore > 0:
                simTotal += simScore
                weightedSum += simScore * rating
        if simTotal == 0:
            raise PredictionImpossible('No neighbors')
        predictedRating = weightedSum / simTotal
        return predictedRating

# Load the model from the pickle file
with open('content_knn_model.pkl', 'rb') as f:
    contentKNN = pickle.load(f)

# Convert model attributes to dictionary
model_data = {
    "recipes": {key: value.to_dict() for key, value in contentKNN.recipes.items()},
    "recipes_index": contentKNN.recipes_index,
    "similarities": contentKNN.similarities.tolist()
}

# Save model data to JSON file
with open('content_knn_model.json', 'w') as f:
    json.dump(model_data, f)
