import click
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@click.command()
@click.option("--name", "-n",
               help="Name of Person",
               required=True,
               prompt="Your name, please?\n")
def supercli(name):
    '''
    This prints "Hello, {name}"
    '''
   
    movies = pd.read_csv("data.txt")
    movies["Clean_Rating"] = movies["Rating"].str.replace("Five", "5", regex=True)
    movies["Clean_Rating"] = movies["Clean_Rating"].str.replace(r'[^0-9,.]', '', regex=True)
    movies = movies[movies["Clean_Rating"] != ""]   
    movies["Clean_Rating"] = movies["Clean_Rating"].fillna(0)
    movies = movies.reset_index(drop=True)

    # vectorizer = TfidfVectorizer(ngram_range=(1,2))
    # tfidf = vectorizer.fit_transform(movies["Movie"])

    # def search(title):
    #     # title = "Fiction"
    #     query_vec = vectorizer.transform([title])
    #     similarity = cosine_similarity(query_vec, tfidf).flatten()
    #     indices = np.argpartition(similarity, -5)[-5:]
    #     results = movies.iloc[indices][::-1]
    #     return results
    
    vectorizer2 = TfidfVectorizer(ngram_range=(1,2))
    tfidf2 = vectorizer2.fit_transform(movies["User"])

    def search_user(user):
        query_vec = vectorizer2.transform([user])
        similarity = cosine_similarity(query_vec, tfidf2).flatten()
        indices = np.argpartition(similarity, -5)[-10:]
        results = movies.iloc[indices][::-1]
        return results
    
    movies["Clean_Rating"] = movies["Clean_Rating"].astype(float)

    def user_recommended_movies(user):
        # user_2 = "Frank
        user_watching = movies[(movies["User"] == user) & (movies["Clean_Rating"] >= 3)]
        ## Similar users with ratings over 3 for same movies user likes
        users_watching_similar_movies = movies[(movies["Movie"].isin(user_watching["Movie"])) & (movies["Clean_Rating"] >= 3)]
        # ## We have to find other movies these users have liked that have ratings of >= 3
        similar_user_recs = movies[(movies["User"].isin(users_watching_similar_movies["User"])) & (movies["Clean_Rating"] >= 3)][["Movie", "User"]]
        user_recs = similar_user_recs[(similar_user_recs["User"] != user)]["Movie"].unique()
        # similar_user_recs
        user_recs_list = user_recs.tolist()
        
        return user_recs_list
    
    movie_list = movies["Movie"].unique()
    movie_list = movie_list.tolist()
    
   
    def find_similar_movie(movie):
        #Finding recommendations from similar users
        similar_users = movies[(movies["Movie"] == movie) & (movies["Clean_Rating"] >= 3)]["User"].unique()
        similar_user_recs = movies[(movies["User"].isin(similar_users)) & (movies["Clean_Rating"] >= 3)]["Movie"]
        
        #Only 10% of users
        similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
        similar_user_recs = similar_user_recs[similar_user_recs > .10]

        #How common the recommendations were amongst other users
        # all_users = movies[(movies["Movie"].isin(similar_user_recs.index)) & (movies["Clean_Rating"] >= 3)]
        # all_user_recs = all_users["Movie"].value_counts() / len(all_users["User"].unique())

        # rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
        # rec_percentages.columns = ["similar", "all"]
            
        # rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
        # rec_percentages = rec_percentages.sort_values("score", ascending=False)

        return similar_user_recs
    
    if len(user_recommended_movies(name)) > 0:
        print(user_recommended_movies(name))
    else:
        for mo in movie_list:
            print(find_similar_movie(mo))