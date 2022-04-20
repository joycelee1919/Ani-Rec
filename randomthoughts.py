import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
import string
import nltk
nltk.download('stopwords')
from collections import Counter

from PIL import Image
import requests
from IPython.display import display

from difflib import get_close_matches

from sklearn.metrics.pairwise import cosine_similarity

from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
import streamlit as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#------------------------------

st.title('‧͙⁺˚*･༓☾ AniRec Randomthoughts ✿✼:*ﾟ:༅｡')

#------------------------------

stpwrd = nltk.corpus.stopwords.words('english')

numbers = map(str, range(1001))

alphabet = list(string.ascii_lowercase)

stpwrd += ['old', 'new', 'however', 'must', 'action', 'adventure', 'comedy', 'drama', 
'sci-fi', 'space', 'mystery', 'shounen', 'police', 'supernatural', 
'magic', 'fantasy', 'sports', 'josei', 'romance', 'slice', 'body', 'every',
'cars', 'seinen', 'horror', 'psychological', 'thriller', 'super power', 
'martial arts', 'school', 'ecchi', 'vampire', 'military', 'historical', 'stop',
'dementia', 'mecha', 'demons', 'samurai', 'game', 'shoujo', 'harem', 'music', 
'shoujo', 'kids', 'hentai', 'parody', 'yuri', 'yaoi', 'leaves', 'along',
'way', 'year', 'source', 'ann', 'one', 'two', 'three', 'four', 'five',
'six', 'seven', 'eight', 'nine', 'ten', 'zero', 'hundred', 'thousand',
'back', 'many', 'named', 'many', 'much', 'different', '—', 'high', 'decides',
'set', 'day', 'episode', 'takes', 'aired', 'years', '1st', '2nd', '3rd',
'dvd', 'becoming', 'go', 'part', 'season', 'upon', 'without', 'with', 'since',
'ova', 'ona', 'face', 'make', 'come', 'seems', 'middle', 'become', 'episodes',
'...', 'hile', 'let', '",', 'causes', 'certain', 'main', 'first', 'see', 'sees'] 

stpwrd += numbers
stpwrd += alphabet

#------------------------------

shounen = Image.open(requests.get("https://wallpaperaccess.com/full/39033.png", stream=True).raw) #Action, Shounen, Mecha
family = Image.open(requests.get("https://funimation.in/wp-content/uploads/2022/04/Top-10-Anime-Similar-To-Spy-X-Family-That-You.jpg", stream=True).raw) #Slice of Life, Comedy, School
ghibli = Image.open(requests.get("https://animeaddictweb.files.wordpress.com/2016/10/ghibli-collage.jpg", stream=True).raw) #Adventure
school = Image.open(requests.get("https://gaijinpot.scdn3.secure.raxcdn.com/app/uploads/sites/4/2018/09/featuredkiminonawa.jpg", stream=True).raw) #Romance
scifi = Image.open(requests.get("https://mangathrill.com/wp-content/uploads/2020/12/pjimage-2020-12-03T192437.985.jpg", stream=True).raw) #Sci-Fi, Thriller, Mystery
fantasy = Image.open(requests.get("https://i.ytimg.com/vi/7svYxJJtC3M/maxresdefault.jpg", stream=True).raw) #Magic, Fantasy
allanime = Image.open(requests.get("https://www.animationmagazine.net/wordpress/wp-content/uploads/Shinkai-Collection.jpg", stream=True).raw) #else

def display_image(sorted_df):

    first_genre = sorted_df['Genres'].iloc[0].split(',')

    if any(word in first_genre for word in ['Action', 'Shounen', 'Mecha']):
        st.image(shounen) 
    elif any(word in first_genre for word in ['Slice of Life', 'Comedy', 'School', 'Seinen']):
        st.image(family) 
    elif any(word in first_genre for word in ['Adventure', 'Kids']):
        st.image(ghibli) 
    elif any(word in first_genre for word in ['Romance', 'Drama', 'Shoujo']):
        st.image(school) 
    elif any(word in first_genre for word in ['Sci-Fi', 'Thriller', 'Mystery', 'Space', 'Police']):
        st.image(scifi) 
    elif any(word in first_genre for word in ['Magic', 'Fantasy', 'Supernatural', 'Super Power']):
        st.image(fantasy) 
    else:
        st.image(allanime) 

#------------------------------

kw_url_og = "https://docs.google.com/spreadsheets/d/1e2bBx9ImdlIav45zxGWxu-jTMH3SWUmpBYGOXYSY3bE/edit#gid=839328658"
kw_url = kw_url_og.replace("/edit#gid=", "/export?format=csv&gid=")
anime_kw = pd.read_csv(kw_url)

anime_kw_syn100 = anime_kw[(anime_kw.Synopsis.str.len() > 100)] 
cv = CountVectorizer()

#------------------------------

brainstorm = st.text_input("What's in your mind?")

#------------------------------

surprise_me = st.button("Enter")

if surprise_me: 
    keywords_matrix = cv.fit_transform(anime_kw_syn100.Keywords) 
    brainstorm_matrix = cv.transform(np.array([brainstorm])) 

    cosine_sim = cosine_similarity(keywords_matrix, brainstorm_matrix)

    cosine_sim_flattened = np.ravel(cosine_sim)

    top5_index = np.argpartition(-cosine_sim_flattened, 5)
    top5_index = top5_index[:5]

    top5_simscore = np.partition(-cosine_sim_flattened, 5)
    top5_simscore = -top5_simscore[:5]

    brainstorm_df = anime_kw_syn100[['MAL_ID', 'Name', 'English_name', 'Japanese_name',
                            'Genres', 'Synopsis', 'Type', 'Episodes', 
                            'Rating', 'Polarity', 'Keywords']].copy()


    brainstorm_df['Similarity_score'] = pd.Series(cosine_sim_flattened)
    brainstorm_df['Similarity_score'] = brainstorm_df['Similarity_score'].apply(lambda x:round(x,3)) 

    top1, top2, top3, top4, top5 = top5_index

    brainstorm_df_output = pd.concat([brainstorm_df.iloc[[top1]], brainstorm_df.iloc[[top2]], 
                        brainstorm_df.iloc[[top3]], brainstorm_df.iloc[[top4]], 
                        brainstorm_df.iloc[[top5]]], ignore_index=True)

    display_df = brainstorm_df_output.drop('Keywords', axis = 1)

    elements = [word for word, word_count in Counter(" ".join(brainstorm_df_output["Keywords"]).split()).most_common(30) if word not in stpwrd ]
    st.write("\nYou entered: " + brainstorm)
    st.write("\nWe think you'd like anime with these elements: " + str(elements))

    sorted_df = display_df.sort_values('Similarity_score', ascending = False)

    display_image(sorted_df)

    st.write("Our top 5 recommendations for you:\n")
    
    gb = GridOptionsBuilder.from_dataframe(sorted_df)
    gb.configure_default_column(editable=False, resizable=True,sorteable=True)
    
    AgGrid(sorted_df, gridOptions = gb)
    
#------------------------------
