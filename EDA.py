#!/usr/bin/env python
# coding: utf-8

# # Primero, comenzamos importando nuestros dataframes, ya modificados anteriormente, para poder comenzar a realizar nuestro Analisis Exploratorio de los Datos:

# Realizamos la importancion de las librerias que vamos a utilizar para el EDA (Exploratory Data Analysis) y también lo que vamos a necesitar a posteriorir para desarrollar el modelo de recomendacion.

# In[106]:


import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel




# In[2]:


# Cargamos los DataFrames desde archivos CSV
df_games = pd.read_csv('df_games.csv')
df_reviews = pd.read_csv('df_reviews.csv')
merged_df2 = pd.read_csv('merged_df2.csv')

# Cargamos el DataFrame desde archivo Parquet
df_items = pq.read_table('df_items.parquet').to_pandas()


# En el archivo Proyecto1.py ya verificamos nulos y duplicados y procedimos a eliminarlos para poder trabajar correctamente con nuestros datos. Ahora verificamos nuevamente.

# In[3]:


df_reviews.info()


# In[4]:


merged_df2 = merged_df2.dropna()


# In[5]:


merged_df2.info()


# In[40]:


df_items.info()


# In[7]:


df_games.info()


# In[8]:


df_games.isna().sum()


# In[9]:


# Como verificamos que existen valores nulos en las diferentes columnas, vamos a proceder a eliminarlos para poder trabajar con nuestro modelo:
df_games = df_games.dropna()


# In[10]:


# Verificamos:
df_games.info()


# In[11]:


df_games.head()


# Una vez verificado esto, continuamos con nuestro Análisis Exploratorio de datos.
# Comenzamos con el dataframe 'df_games': 

# In[30]:


# Queremos hacer un histograma de precios, pero como verificamos, la columna price no está en formato numerico y tiene valores que no son numeros,
# por lo que en primer lugar debemos cambiarla:

df_games.head()


# In[16]:


df_games['price'] = pd.to_numeric(df_games['price'], errors='coerce').fillna(0)

# Con esto modificamos los juegos que son gratis ('free to play) o simplemente estan bonificados, pasando esos registros a valor 0 y quedandonos con los que realmente tienen precio.


# In[28]:


# # Aplicamos una transformación logarítmica a la columna 'price' antes de crear el histograma para obtener una mejor visualización de la distribución.

df_games['price_log'] = np.log1p(df_games['price'])  # Agrega una pequeña constante para evitar log(0)

# Creamos el histograma para visualizar: 

plt.figure(figsize=(10, 6))
sns.histplot(df_games['price_log'], bins=30, kde=True)
plt.title('Distribución de Precios de Videojuegos (escala logarítmica)')
plt.xlabel('Log(Precio)')
plt.ylabel('Frecuencia')
plt.show()


# Verificamos que la mayoria de los precios estan entre $0.00 (juegos que son gratis, estan bonificados, etc) y $4.00. 

# In[36]:


# Procedemos a verificar la cantidad de generos y cuales son los mas repetidos: 

# Dividimos primero la columna 'genres' en listas y 'explotamos' (split y expand) las listas en filas individuales: 
genres_series = df_games['genres'].str.split(',').explode().str.strip()

# Contamos la frecuencia de géneros únicos:
genre_counts = genres_series.value_counts()

# Filtramos géneros con una frecuencia mínima para mantener la legibilidad, en este caso al tener +25.000 registros, establecemos el minimo en 1000,
# con el objetivo de solamente traernos los generos que mas se repiten: 
min_frequency = 1000
filtered_genre_counts = genre_counts[genre_counts >= min_frequency]

plt.figure(figsize=(12, 6))
sns.barplot(x=filtered_genre_counts.index, y=filtered_genre_counts.values, palette='viridis')
plt.title('Frecuencia de Géneros de Videojuegos')
plt.xlabel('Género')
plt.ylabel('Frecuencia')
plt.xticks(rotation=90)
plt.show()


# Pasamos a analizar el dataframe 'df_items' : 

# In[45]:


# Al tener muchos juegos que con valor de 0 tiempo jugado, filtramos los juegos con tiempo de juego mayor que 0, que es lo que realmente nos interesa:
df_items_filtered = df_items[df_items['playtime_forever'] > 0].copy()

# Al igual que hicimos anteriormente, realizamos una transformación logarítmica al tiempo de juego (playtime_forever) para poder mostrar correctamente la informacion:
df_items_filtered['playtime_log'] = np.log1p(df_items_filtered['playtime_forever'])  # Agregamos una pequeña constante para evitar log(0)

# Realizamos la distribucion: 
plt.figure(figsize=(10, 6))
sns.histplot(df_items_filtered['playtime_log'], bins=30, kde=True)
plt.title('Distribución del Tiempo de Juego (Escala Logarítmica)')
plt.xlabel('Log(Tiempo de Juego)')
plt.ylabel('Frecuencia')
plt.show()


# In[52]:


# Ahora realizamos un box plot (grafico de caja). Este código de abajo utiliza los datos filtrados en df_items_filtered que excluyen los juegos con tiempo de juego igual a 0 y 
# muestran la distribución del tiempo de juego en un gráfico de caja con la escala logarítmica aplicada. 
# El gráfico de caja es importante para poder visualizar información sobre la mediana, los cuartiles y los valores atípicos en la distribución del tiempo de juego:

plt.figure(figsize=(10, 6))
sns.boxplot(y='playtime_log', data=df_items_filtered)
plt.title('Distribución del Tiempo de Juego (Escala Logarítmica) - Box Plot')
plt.ylabel('Log(Tiempo de Juego)')
plt.show()


# Analizamos detalladamente este gráfico
# 
# En un gráfico de caja (box plot), los elementos clave a considerar son el "cuadro azul", las "líneas" y los "puntos" que están encima, en la parte superior:
# 
# Cuadro Azul (Box):
# El cuadro azul en el medio del gráfico de caja representa el rango intercuartil (IQR, por sus siglas en inglés), que es la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1). El cuadro abarca el 50% central de los datos.
# La línea que atraviesa el cuadro azul es la mediana (Q2), que representa el valor medio de la distribución.
# 
# Líneas (Bigotes):
# Las "líneas" que se extienden desde el cuadro azul son los llamados "bigotes". Generalmente, se extienden desde el primer cuartil (Q1) hasta el valor más bajo dentro de 1.5 veces el IQR por debajo de Q1, y desde el tercer cuartil (Q3) hasta el valor más alto dentro de 1.5 veces el IQR por encima de Q3.
# Los valores que están fuera de esta distancia (más allá de los bigotes) se consideran valores atípicos y se representan como "puntos" en el gráfico.
# 
# Puntos en la parte superior (Outliers):
# Los puntos que están por encima o por debajo de los bigotes se consideran valores atípicos o outliers. Estos son valores que están significativamente más altos o más bajos que la mayoría de los datos en la distribución.
# Los valores atípicos se muestran como puntos individuales en el gráfico y pueden indicar la presencia de datos inusuales o extremos.
# En resumen, el gráfico de caja proporciona una representación visual de la distribución de los datos, mostrando la mediana, el rango intercuartil y la presencia de valores atípicos. El cuadro azul y los bigotes delimitan el rango intercuartil, mientras que los puntitos indican valores que pueden ser inusuales en comparación con el resto de los datos. Esto es útil para identificar la dispersión y la presencia de valores extremos en la variable representada en el gráfico.
# 
# 
# 
# 
# 
# 

# In[61]:


# Ahora, realizamos un gráfico de barras con la columna item_name (nombre de elementos), para verificar los juegos que mas aparecen en el dataframe:

plt.figure(figsize=(12, 6))
item_name_counts = df_items['item_name'].value_counts().head(20)  # Aca podemos ajustar la cantidad de elementos mostrados, en este caso mostramos un top 20
sns.barplot(x=item_name_counts.values, y=item_name_counts.index, palette='viridis')
plt.title('Frecuencia en la que aparece cada Juego (item_name - Top 20)')
plt.xlabel('Frecuencia')
plt.ylabel('Nombre del Elemento')
plt.xticks(rotation=0)  # Aca podemos ajustar la rotación de las etiquetas del eje de frecuencia si es necesario
plt.show()


# Pasamos a analizar el dataframe 'df_reviews' : 

# In[62]:


df_reviews.head()


# In[63]:


# Creamos un DataFrame parcial que cuente la proporción de recomendaciones positivas y negativas
recommendation_counts = df_reviews['recommend'].value_counts(normalize=True)

# Creamos un gráfico de barras apiladas en funcion de las recomendaciones de los juegos, para verificar si hay mas juegos recomendados, que no recomendados:
plt.figure(figsize=(8, 6))
sns.barplot(x=recommendation_counts.index, y=recommendation_counts.values, palette='viridis')
plt.title('Proporción de Recomendaciones en Revisiones de Juegos')
plt.xlabel('Recomendación')
plt.ylabel('Proporción')
plt.show()


# In[82]:


# Podemos tambien visualizarlo mas facilmente con un grafico de torta de Recomendaciones (pie plot):

plt.figure(figsize=(6, 6))
recommendation_labels = ['Recomendado', 'No Recomendado']
plt.pie(recommendation_counts, labels=recommendation_labels, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral'])
plt.title('Proporción de Recomendaciones respecto al total de Reviews de los Juegos')
plt.show()


# In[65]:


# También realizamos un Histograma de Análisis de Sentimiento (sentiment_analysis):
# Creamos un mapeo de valores a etiquetas:
sentiment_labels = {
    0: 'Recomendación Mala',
    1: 'Recomendación Neutra',
    2: 'Recomendación Positiva'
}

# Creamos el gráfico de barras para el análisis de sentimiento:
plt.figure(figsize=(10, 6))
sns.histplot(df_reviews['sentiment_analysis'], bins=[-0.5, 0.5, 1.5, 2.5], kde=True, discrete=True)
plt.title('Distribución del Análisis de Sentimiento en Revisiones de Juegos')
plt.xlabel('Análisis de Sentimiento')
plt.ylabel('Frecuencia')
plt.xticks([0, 1, 2], [sentiment_labels[i] for i in [0, 1, 2]])  # Etiquetas personalizadas en el eje x
plt.show()



# In[86]:


df_games.head()


# In[167]:


df_reviews.tail(20)


# In[ ]:





# # Una vez realizado nuestro Analisis Exploratorio y según las conclusiones obtenidas, realizamos nuestro modelo de recomendaciones de juegos: 

# Procedemos a realizar el sistema de recomendacion. De las dos opciones que nos dieron, optamos por realizar  una relación ítem-ítem.
# Esto significa que le pasamos el item de un juego al modelo y, en base a que tan similar esa ese ítem al resto, el modelo nos recomienda juegos similares. 
# El input es un juego (su id) y el output es una lista de 5 juegos recomendados. Para ello, aplicamos como base la similitud del coseno.

# In[227]:


def recomendacion_juego3(item_id):
    # Número de recomendaciones (ajustable según lo que se prefiera) en este caso ponemos "6" para traer 5 recomendaciones omitiendo la primera, 
    # ya que siempre nos va a traer el mismo juego primero, por una cuestion propia del funcionamiento interno de la similitud del coseno:
    num_recommendations = 6

    # Nos basamos en los tags y géneros del juego de referencia: 
    reference_game = df_games[df_games['id'] == item_id]
    reference_tags = reference_game['tags'].values[0]
    reference_genres = reference_game['genres'].values[0]

    # Combinamos los tags y géneros en una sola cadena de texto:
    reference_features = ' '.join([reference_tags, reference_genres])

    # Se rellenan valores NaN en las columnas 'tags' y 'genres' con cadenas vacías, para evitar errores
    df_games['tags'] = df_games['tags'].fillna('')
    df_games['genres'] = df_games['genres'].fillna('')

    # Creo un vectorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()

    # Se aplica el vectorizador a todas las características de los juegos, para poder realizar las
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_games['tags'] + ' ' + df_games['genres'])

    # Calculo la similitud de coseno entre las características del juego de referencia y los otros juegos
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_vectorizer.transform([reference_features]))

    # Obtenemos los índices de los juegos más similares
    similar_game_indices = cosine_similarities.argsort(axis=0)[:-num_recommendations-1:-1]

    # Obtenemos los nombres de los juegos recomendados a partir de la segunda recomendación
    recommended_game_names = df_games.iloc[similar_game_indices.flatten()][1:num_recommendations + 1]['title'].tolist()

    return recommended_game_names


# In[229]:


# Utilizamos un ID aleatorio para probar que el modelo funcione:
product_id = 130  
recommended_games = recomendacion_juego3(product_id)
print(recommended_games)


# Efectivamente, verificamos que el sistema funciona. Le pasamos un juego y nos devuelve juegos similares.
