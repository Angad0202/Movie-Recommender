from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__, static_url_path='', static_folder='templates')

# Load movie data
movies_df = pd.read_csv("movies.csv")
movies_df.drop_duplicates(subset=['title','release_date'], inplace=True)
movies_df = movies_df[movies_df['vote_count'] >= 20].reset_index(drop=True)
movies_df.fillna('', inplace=True)
movies_df.dropna(subset=['genres', 'overview'], inplace=True)
movies_df['tags'] = movies_df['overview'] + ' ' + movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['original_language']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['tags'])

# Helper function to get the index of a movie
def get_movie_index(title):
    return movies_df.index[movies_df['title'] == title][0]

# route for styles.css
@app.route('/styles.css')
def css():
    return send_from_directory(app.static_folder, 'styles.css')

# Route for the home page
@app.route("/")
def home():
    return render_template('home.html')

# Route for the recommendations page
@app.route("/get_recommendations", methods=["POST"])
def get_recommendations():
    title = request.form["title"]
    if title not in list(movies_df['title']):
        return render_template('home.html', message="Movie not found")
    else:
        movie_index = get_movie_index(title)
        sim_scores = list(enumerate(cosine_similarity(tfidf_matrix, tfidf_matrix[movie_index])))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = list(movies_df.iloc[movie_indices]['title'])
        return render_template('home.html', recommendations=recommended_movies, title="Recommendations for "+title)

if __name__ == '__main__':
    app.run(debug=True)
