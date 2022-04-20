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

import streamlit as st
st.set_page_config(layout="wide")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#------------------------------

kw_url_og = "https://docs.google.com/spreadsheets/d/1e2bBx9ImdlIav45zxGWxu-jTMH3SWUmpBYGOXYSY3bE/edit#gid=839328658"
kw_url = kw_url_og.replace("/edit#gid=", "/export?format=csv&gid=")
anime_kw = pd.read_csv(kw_url)

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
        st.image(shounen, width = 600) 
    elif any(word in first_genre for word in ['Slice of Life', 'Comedy', 'School', 'Seinen']):
        st.image(family, width = 600) 
    elif any(word in first_genre for word in ['Adventure', 'Kids']):
        st.image(ghibli, width = 600) 
    elif any(word in first_genre for word in ['Romance', 'Drama', 'Shoujo']):
        st.image(school, width = 600) 
    elif any(word in first_genre for word in ['Sci-Fi', 'Thriller', 'Mystery', 'Space', 'Police']):
        st.image(scifi, width = 600) 
    elif any(word in first_genre for word in ['Magic', 'Fantasy', 'Supernatural', 'Super Power']):
        st.image(fantasy, width = 600) 
    else:
        st.image(allanime, width = 600) 

#------------------------------

def search_casein(string_entry, df):

    # By name
    searched_ci_name = df[df['Name'].str.contains(f"(?i){string_entry}")]
    searched_ci_eng = df[df['English_name'].str.contains(f"(?i){string_entry}")]
    searched_ci_jp = df[df['Japanese_name'].str.contains(f"(?i){string_entry}")]

    # By MAL_ID
    df['MAL_ID'] = df['MAL_ID'].astype("str")
    searched_ID = df[df['MAL_ID'] == string_entry]
    
    return searched_ci_name, searched_ci_eng, searched_ci_jp, searched_ID

def fuzzy_search(string_entry, df):

    fuzzy_name = get_close_matches(string_entry, df["Name"], n = 5, cutoff = 0.5)
    fuzzy_eng = get_close_matches(string_entry, df["English_name"], n = 5, cutoff = 0.5)
    fuzzy_jap = get_close_matches(string_entry, df["Japanese_name"], n = 5, cutoff = 0.5)
    #The best (no more than n) matches among the possibilities are returned in a list, sorted by similarity score, most similar first.

    return fuzzy_name, fuzzy_eng, fuzzy_jap

#------------------------------

cv = CountVectorizer()

#------------------------------

st.caption("WARNING: streamlit runs like a snail - please be patient💖")

st.title('‧͙⁺˚*･༓☾ AniRec engine made with L0VE ✿✼:*ﾟ:༅｡')

st.image("https://animesher.com/orig/1/133/1331/13313/animesher.com_gif-funny-himouto-umaruchan-1331344.gif", width=400)

# Option RADIO BUTTONS x3 cuz buttons don't work -_-

status = st.radio("How would you like to get recommended today?", 
                  ("✨Let Your Mind Roam Free✨","Based on an anime you love 🥰", "What do other people think? 💭"))  

#------------------------------

if status == "✨Let Your Mind Roam Free✨": # brainstorm plot elements
    
    st.header("Random thoughts")
    st.write("Write down ANYTHING you like, any plot element, any words, any phrases!")
    st.caption("e.g., time travel love death memory sea flower")

    #------------------------------

    anime_kw_syn100 = anime_kw[(anime_kw.Synopsis.str.len() > 100)] 

    #------------------------------

    brainstorm = st.text_input("What's in your mind?")

    #------------------------------

    start_random_thoughts = st.button("Enter")

    if start_random_thoughts: 
        
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

        st.table(sorted_df)

#------------------------------

elif status == "Based on an anime you love 🥰": # content-based
        
    st.header('Content-based recommendation')
    st.write("Don't worry if you don't know the exact title, suggestions will be given :3")
    st.caption("Try entering random stuff like: hOwl'Ss m0vingggu c4stleeeeee, LapUTa, もののけ姫 and see what happens?")

    #------------------------------

    string_entry = st.text_input("\nEnter an anime title/MAL_ID to get recommended :D") 

    #------------------------------

    content_rec_start = st.button("Enter")

    if content_rec_start: 

        searched_ci_name, searched_ci_eng, searched_ci_jp, searched_ID = search_casein(string_entry, anime_kw)

        search_str_result = pd.concat([searched_ci_name, searched_ci_eng, searched_ci_jp, searched_ID], ignore_index=True)
        search_str_result = search_str_result.drop_duplicates(subset='MAL_ID', keep="first")
        search_str_result = search_str_result.drop('Keywords', axis = 1)

        if len(search_str_result) == 0:
            st.write("\nNo matches found :( Do you mean:\n")

            fuzzy_name, fuzzy_eng, fuzzy_jap = fuzzy_search(string_entry, anime_kw)
            combined_suggestions = fuzzy_name + fuzzy_eng + fuzzy_jap

            if len(set(combined_suggestions)) == 0:
                st.write("----Sorry - no suggestion available. Please try again!----")

            else:
                st.write(set(combined_suggestions), sep = '\n')

        elif len(search_str_result) > 1:
            st.write("\nMultiple results found :) Please select a unique MAL_ID from below for a more precise recommendation:\n")
            st.table(search_str_result[['MAL_ID', 'Name', 'English_name', 'Japanese_name', 'Type', 'Episodes']].head(10))

        else:
            st.write("\nAnime selected:")
            st.table(search_str_result)

            # Obtain index of searched anime for later matching
            # 'Name' is unique
            indices = pd.Series(anime_kw.index, index = anime_kw['Name']) # Name is unique in og df
            title = search_str_result['Name'].iloc[0] # Match with that in selected result
            idx = indices[title]

            # Start recommending based on keyword similarity
            keywords = cv.fit_transform(anime_kw['Keywords'])
            cosine_sim_keywords = cosine_similarity(keywords, keywords)

            # Create new df display (rec based on keyword similarity)
            kw_rec_df = anime_kw[['MAL_ID', 'Name', 'English_name', 'Japanese_name',
                                    'Genres', 'Type', 'Episodes', 'Synopsis', 
                                    'Rating', 'Polarity', 'Keywords']].copy()  # Keep 'Keywords' for identifying elements

            # Similarity_column: contains cosine similarity scores of selected anime VS each of the other anime
            kw_rec_df['Similarity_score'] = pd.Series(cosine_sim_keywords[idx])
            kw_rec_df['Similarity_score'] = kw_rec_df['Similarity_score'].apply(lambda x: round(x, 3)) # 3 dp not so precise in display
            kw_rec_df = kw_rec_df.sort_values(['Similarity_score', 'Rating'], ascending = False).iloc[1:11]            

            # Final output - discard Keyword
            display_df = kw_rec_df.drop('Keywords', axis = 1)

            # Display common elements and results

            elements = [word for word, word_count in Counter(" ".join(kw_rec_df["Keywords"]).split()).most_common(30) if word not in stpwrd ]
            st.write("\nWe think you'd like anime with these elements: " + str(elements))
            st.write("\nGenres that you like:", search_str_result['Genres'].iloc[0])

            display_image(search_str_result)

            st.write("\nSo here are our top 10 recommendations for you:")
            st.table(display_df)

#------------------------------

elif status == "What do other people think? 💭": # collaborative user rating clustering
    
    st.balloons()
    st.header('Collaborative user rating clustering')
    st.write("TOO BAD. There's nothing in here D:")
    st.write("Streamlit can't handle my lovely huge dataset!!! See you in my m0nStER PC 🔥🔥🔥🔥🔥")

#------------------------------
