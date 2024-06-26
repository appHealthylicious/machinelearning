# Machine Learning for Recipe Recommendations

![RecipeRecommendation](./images/Screenshot%202024-06-21%20185213.png)

This repository contains the code and resources for a machine learning project focused on generating recipe recommendations based on user input ingredients.

## Project Structure

- `data/`: Contains datasets used for training and evaluation.
- `notebooks/`: Jupyter notebooks with exploratory data analysis and model development.
- `.gitignore`: Specifies files to be ignored by git.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/appHealthylicious/machinelearning.git
   cd machinelearning
    ```

2. Clone the repository:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
    ```bash
   jupyter notebook
    ```

## Usage
- Open the notebooks in the notebooks/ directory to explore the data and model training process.
- Modify and run the notebooks to train the model and generate recommendations based on your own input data.

## Datasets
- `recipes.csv`: Contains recipe information including ingredients, instructions, and nutritional data.
- `ratings.csv`: Contains user ratings for different recipes.

## Machine Learning Models

![Recommended for You Pipeline](./images/Healthylicious_page-0012.jpg)
### Recommended for You
Utilizes K-Nearest Neighbors (KNN) for collaborative filtering to recommend recipes based on user-item interactions and user preferences.

1. User Interactions:

- Users choose disliked ingredients and allergies, which are stored in Firebase.
- Users rate recipes, and these ratings are also stored in Firebase.

2. Collaborative Filtering:

- Calculate the similarity between the data points (recipes) and all training data.
- Recommend recipes with the highest similarity while considering the user's dislikes and preferences.

<br>

![Get Recipes Pipeline](./images/Healthylicious_page-0013.jpg)

### Get Recipes
Employs TF-IDF vectorization and cosine similarity to recommend recipes based on the ingredients available at home.

1. User Inputs:

- Users choose disliked ingredients and allergies, which are stored in Firebase.
- Users input ingredients they have and request recipe generation.

2. Content-Based Filtering:

- Retrieve stored dislikes and allergies from Firebase.
- Apply TF-IDF vectorization on user input ingredients.
- Compute cosine similarity to find top N similar recipes.
- Generate and recommend recipes that do not contain the disliked or allergenic ingredients.

## License
This project is licensed under the MIT License - see the LICENSE file for details.