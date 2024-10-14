# Movie Recommendation System 🎬📽️

This project aims to develop a **Movie Recommendation System** using machine learning techniques such as **TF-IDF** and **Cosine Similarity**. By analyzing the content of movies (e.g., genres, overview, cast), the system recommends similar movies based on user input.

## Data
The dataset used for this project includes the following features:

- **index**: Row number or unique index for the dataset.
- **budget**: The budget of the movie in US dollars.
- **genres**: The genres of the movie (e.g., Action, Comedy, Drama).
- **homepage**: URL of the movie's official homepage.
- **id**: Unique identifier for each movie.
- **keywords**: Important keywords associated with the movie.
- **original_language**: The original language in which the movie was produced.
- **original_title**: The original title of the movie.
- **overview**: A brief summary or overview of the movie's plot.
- **popularity**: Popularity score of the movie.
- **production_companies**: The companies involved in producing the movie.
- **production_countries**: The countries where the movie was produced.
- **release_date**: The release date of the movie.
- **revenue**: The revenue earned by the movie in US dollars.
- **runtime**: The runtime of the movie in minutes.
- **spoken_languages**: Languages spoken in the movie.
- **status**: The current status of the movie (e.g., Released, Post-Production).
- **tagline**: The tagline of the movie.
- **title**: The title of the movie.
- **vote_average**: The average rating of the movie.
- **vote_count**: The total number of votes the movie received.
- **cast**: Main cast of the movie.
- **crew**: Crew members involved in the production.
- **director**: Director of the movie.

> **Note:** Some features, such as `overview` and `genres`, are used to create the recommendation model based on textual content.

## Libraries Used
The following libraries are used in this project:

- **NumPy**: For numerical computations and array manipulations.
- **Pandas**: For handling and processing tabular data.
- **Difflib**: For finding the closest matches to a user’s input.
- **TfidfVectorizer** (from `sklearn.feature_extraction.text`): To convert text data (movie overview and genres) into numerical vectors.
- **Cosine Similarity** (from `sklearn.metrics.pairwise`): To measure the similarity between movies based on TF-IDF vectors.

## Notebooks
This directory contains a Jupyter Notebook (`Movie_Recommendation_System_using_Machine_Learning_with_Python.ipynb`) that walks through the entire process of building the movie recommendation system using content-based filtering.

### Key Steps in the Notebook:
1. **Data Loading and Exploration:**
   - Load the dataset and explore basic statistics.
   - Analyze features such as `genres`, `overview`, and `cast` to identify the most useful columns for building the recommendation system.

2. **Text Preprocessing with TF-IDF:**
   - Use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text features such as `genres` and `overview` into numerical representations. 
   - The higher the TF-IDF score, the more relevant the term is to that specific movie.

   ```python
   tfidf_vectorizer = TfidfVectorizer(stop_words='english')
   tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['overview'])
   ```

3. **Computing Cosine Similarity:**

Use **Cosine Similarity** to measure how similar two movies are based on their TF-IDF vectors. Movies with higher similarity scores are recommended to the user.

```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

4. **Building the Recommendation System:**

- Input a movie title and find the most similar movies based on their content (e.g., genres, plot).
- Use **difflib** to find the closest match for user input in case of misspellings or partial titles.

```python
difflib.get_close_matches(movie_title, movie_titles_list)
```

5. **Model Evaluation:**

- The model performance is evaluated based on user satisfaction and the accuracy of recommendations.
- Use test cases to verify that the system recommends movies that are indeed similar in content.

6. **Visualization of Results:**

- Visualize the top recommended movies for a given input.
- Optionally, analyze the impact of different features like genres and overview on the similarity scores.

## Customization:

Modify the Jupyter Notebook to:
- Experiment with different preprocessing techniques, such as Lemmatization or Stemming in text data.
- Use other text-based similarity measures like Euclidean Distance or Jaccard Similarity.
- Incorporate additional features, such as cast, director, or keywords, to improve the recommendations.

## Resources:

- Sklearn TfidfVectorizer Documentation: [https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Sklearn cosine_similarity Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- Movie Dataset: [https://drive.google.com/file/d/1KKj4-tT2wtGhc6T7LiETG2dTy79aPUiA/view?usp=drive_link](https://drive.google.com/file/d/1KKj4-tT2wtGhc6T7LiETG2dTy79aPUiA/view?usp=drive_link)
> **Note:** You need to request access to view the complete file.
## Further Contributions:

Extend this project by:
- Adding a user interface for real-time movie recommendations.
- Implementing a hybrid recommendation system by incorporating collaborative filtering with user data (ratings, watch history).
- Exploring explainability techniques to provide users with insights into why certain movies are recommended.

By leveraging machine learning models like TF-IDF and Cosine Similarity, this project aims to develop a content-based movie recommendation system. It serves as a foundation for further exploration into personalized recommendation systems in the entertainment industry.
