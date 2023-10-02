#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyarrow.parquet as pq
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



# In[2]:


# Cargar los DataFrames desde archivos CSV
df_games = pd.read_csv('df_games.csv')
df_reviews = pd.read_csv('df_reviews.csv')
merged_df2 = pd.read_csv('merged_df2.csv')

# Cargar el DataFrame desde archivo Parquet
df_items = pq.read_table('df_items.parquet').to_pandas()


# In[3]:


df_games.info()


# In[4]:


df_games.isna().sum()


# In[5]:


# Como verificamos que existen valores nulos en las diferentes columnas, vamos a proceder a eliminarlos para poder trabajar con nuestro modelo:
df_games = df_games.dropna()


# In[6]:


# Verificamos:
df_games.info()


# In[11]:


df_games.head(20)


# In[8]:


''' def recomendacion_juego(id_juego):
    # Encuentra el índice del juego en df_games
    juego_index = df_games[df_games['id'] == id_juego].index[0]
    
    # Obtén la fila de similitud del juego con otros juegos
    juego_similaridad = similarity_matrix[juego_index]
    
    # Ordena los juegos por similitud y selecciona los 5 principales (excluyendo el juego de entrada)
    juegos_similares_indices = juego_similaridad.argsort()[::-1][1:6]
    
    # Obtiene los títulos de los juegos similares
    juegos_similares = df_games.iloc[juegos_similares_indices]['title'].tolist()
    
    return juegos_similares

'''


# In[20]:


def recomendacion_juego(product_id):
    # Especifica el tamaño de la muestra
    tamano_muestra = 1200
    
    # Toma una muestra aleatoria del DataFrame original
    muestra_aleatoria = df_games.sample(n=tamano_muestra, random_state=42)
    
    # Reemplaza con el product_id proporcionado por el usuario
    target_game = muestra_aleatoria.copy()
    target_game['id'] = product_id

    # Combina las etiquetas (tags) y géneros en una sola cadena de texto
    target_game_tags_and_genres = ' '.join(target_game['tags'].fillna('').astype(str) + ' ' + target_game['genres'].fillna('').astype(str))

    # Crea una lista que contenga el juego de referencia y todos los juegos en la muestra
    all_games = [target_game_tags_and_genres] + muestra_aleatoria['tags'].astype(str) + ' ' + muestra_aleatoria['genres'].astype(str)

    # Crea un vectorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()

    # Aplica el vectorizador a la lista de juegos
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_games)

    # Calcular la similitud entre el juego de referencia y todos los demás juegos
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Ordenar los juegos por similitud y obtener los índices de los juegos más similares
    similar_games_indices = similarity_matrix[0].argsort()[::-1]

    # Recomendar los juegos más similares (puedes ajustar el número de recomendaciones)
    num_recommendations = 5
    recommended_games = muestra_aleatoria.iloc[similar_games_indices[1:num_recommendations + 1]]

    # Devuelve la lista de juegos recomendados
    recommended_games_list = recommended_games['title'].tolist() 
    
    return recommended_games_list


# In[21]:


resultado = recomendacion_juego(643980)
resultado

