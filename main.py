import tkinter as tk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer




movies_df = pd.read_csv("movies.csv")

movies_df.drop_duplicates(subset=['title','release_date'], inplace=True)

movies_df = movies_df[movies_df['vote_count'] >= 20].reset_index(drop=True)

movies_df.fillna('', inplace=True)

movies_df.dropna(subset=['genres', 'overview'], inplace=True)

movies_df['tags'] = movies_df['overview'] + ' ' + movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['original_language']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['tags'])

def get_movie_index(title):
    return movies_df.index[movies_df['title'] == title][0]

# Get movie recommendations
def get_recommendations(title):
    title = text.get("1.0", "end-1c")
    text2.delete('1.0', tk.END)
    movie_index = get_movie_index(title)
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix, tfidf_matrix[movie_index])))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]
    
    char_list=list(movies_df.iloc[movie_indices]['title'].to_string(index=False))
    movie_name = ""
    movies_list = []
    for char in char_list:
        if char == '\n':
            movies_list.append(movie_name.strip())
            movie_name = ""
        elif char != ' ':
            movie_name += char
        else:
            movie_name += " "
    movies_list.append(movie_name.strip())
    for i in range(len(movies_list)):
        text2.insert(tk.END,f"{i+1}. {movies_list[i]}\n")



root = tk.Tk()
root.title("Movie Recommendation")
root.geometry("720x600")

text = tk.Text(root, height=1, width=40)
text.place(x=230,y=10)
text2 = tk.Text(root, height=15, width=40)
text2.place(x=230,y=150)
title = text.get("1.0", "end-1c")

recommend = tk.Button(root, text="Recommend", command=lambda: get_recommendations(title))
recommend.place(x=300,y=50)

root.mainloop()