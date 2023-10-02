from fastapi import FastAPI
import pandas as pd
import pyarrow.parquet as pq
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Cargar los DataFrames desde archivos CSV
df_games = pd.read_csv('df_games.csv')
df_reviews = pd.read_csv('df_reviews.csv')
merged_df2 = pd.read_csv('merged_df2.csv')

# Cargar el DataFrame desde archivo Parquet
df_items = pq.read_table('df_items.parquet').to_pandas()

app = FastAPI()

# Definir rutas y funciones de ruta aquí

# FUNCIONES:

# Primera funcion:
def PlayTimeGenre(genero, df_games, df_items):
    games_filtered = df_games[df_games['genres'].str.contains(genero, case=False, na=False)]

    # Filter df_items to obtain rows with similar item_id
    merged_df = pd.merge(df_items, games_filtered, left_on='item_id', right_on='id', how='inner')
    merged_df['release_date'] = pd.to_datetime(merged_df['release_date'])
    merged_df['release_year'] = merged_df['release_date'].dt.year

    grouped = merged_df.groupby('release_year')['playtime_forever'].sum()

    max_year = grouped.idxmax()

    return f"Año con más horas jugadas para Género {genero}: {int(max_year)}"

@app.get("/play_time_genre/{genero}")
async def read_play_time_genre(genero: str):
    # Llama a la función PlayTimeGenre y pasa los argumentos necesarios (genero, df_games, df_items)
    resultado = PlayTimeGenre(genero, df_games, df_items)
    
    # Devuelve el resultado como respuesta en formato JSON
    return {"resultado": resultado}

# Segunda Funcion: 

def UserForGenre(genero, df_games, df_items):
    # Filtra los juegos por el género especificado
    games_filtered = df_games[df_games['genres'].str.contains(genero, case=False, na=False)]

    # Fusiona df_items y df_games basado en item_id e id
    merged_df = pd.merge(df_items, games_filtered, left_on='item_id', right_on='id', how='inner')
    merged_df['release_date'] = pd.to_datetime(merged_df['release_date'])
    merged_df['release_year'] = merged_df['release_date'].dt.year

    # Agrupa por año y usuario, calcula las horas jugadas por usuario por año
    grouped = merged_df.groupby(['release_year', 'user_id'])['playtime_forever'].sum().reset_index()

    # Encuentra al usuario con más horas jugadas para el género dado
    max_user = grouped[grouped['playtime_forever'] == grouped.groupby('release_year')['playtime_forever'].transform('max')]['user_id'].values[0]

    # Filtra los datos para el usuario con más horas jugadas
    user_data = grouped[grouped['user_id'] == max_user]

    # Elimina los años con 0 horas jugadas
    user_data = user_data[user_data['playtime_forever'] > 0]

    # Ordena los años en orden descendente
    user_data = user_data.sort_values(by='release_year', ascending=False)

    # Convierte las horas a enteros
    user_data['playtime_forever'] = user_data['playtime_forever'].astype(int)

    # Crea una lista de la acumulación de horas jugadas por año
    hours_by_year = [{'Año': int(year), 'Horas': int(hours)} for year, hours in zip(user_data['release_year'], user_data['playtime_forever'])]

    result = {
        "Usuario con más horas jugadas para Género " + genero: max_user,
        "Horas jugadas": hours_by_year
    }

    return result

@app.get("/user_for_genre/{genero}")
async def read_user_for_genre(genero: str):
    # Llama a la función UserForGenre y pasa los DataFrames como argumentos
    resultado = UserForGenre(genero, df_games, df_items)
    
    # Devuelve el resultado como respuesta en formato JSON
    return {"resultado": resultado} 


# Tercera funcion:

def UsersRecommend(year: int):
    # Asegúrate de que 'posted' sea de tipo datetime
    merged_df2['posted'] = pd.to_datetime(merged_df2['posted'])
    
    # Filtra las reseñas por el año proporcionado
    df_filtered = merged_df2[merged_df2['posted'].dt.year == year]
    
    # Filtra las reseñas recomendadas y con sentimiento positivo o neutral
    df_filtered = df_filtered[(df_filtered['recommend'] == True) & (df_filtered['sentiment_analysis'] >= 1)]
    
    # Cuenta cuántas veces se recomienda cada juego
    top_games = df_filtered['title'].value_counts().head(3)
    
    # Formatea los resultados en el formato deseado
    result = [{"Puesto {}".format(i + 1): game} for i, game in enumerate(top_games.index)]
    
    return result


@app.get("/users_recommend/{year}")
async def read_users_recommend(year: int):
    resultado = UsersRecommend(year)  
    return {"resultado": resultado}


# Cuarta Funcion:

def UsersNotRecommend(year):
    
    merged_df2['posted'] = pd.to_datetime(merged_df2['posted'])

    # Filtra las reseñas por el año proporcionado
    df_filtered = merged_df2[merged_df2['posted'].dt.year == year]

    # Filtra las reseñas no recomendadas y con sentimiento negativo
    df_filtered = df_filtered[(df_filtered['recommend'] == False) & (df_filtered['sentiment_analysis'] == 0)]

    # Cuenta cuántas veces cada juego ha sido no recomendado
    top_not_recommend_games = df_filtered['title'].value_counts().head(3)

    # Formatea los resultados en el formato deseado
    result = [{"Puesto {}".format(i + 1): game} for i, game in enumerate(top_not_recommend_games.index)]
    
    return result

@app.get("/users_not_recommend/{year}")
async def read_users_not_recommend(year: int):
    resultado = UsersNotRecommend(year)
    return {"resultado": resultado}



# Quinta Funcion: 

def SentimentAnalysis(year: int):
    # Asegúrate de que 'posted' sea de tipo datetime
    merged_df2['posted'] = pd.to_datetime(merged_df2['posted']) 
    
    # Filtra las reseñas por el año dado
    reseñas_por_año = merged_df2[merged_df2['posted'].dt.year == year]
    
    # Filtra las reseñas recomendadas y con sentimiento positivo, neutral o negativo
    reseñas_filtradas = reseñas_por_año[reseñas_por_año['recommend'].isin([True, False]) & reseñas_por_año['sentiment_analysis'].isin([0, 1, 2])]
    
    # Mapea los valores de 'sentiment_analysis' a categorías de sentimiento
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    reseñas_filtradas['sentiment_category'] = reseñas_filtradas['sentiment_analysis'].map(sentiment_mapping)
    
    # Cuenta la cantidad de registros en cada categoría de sentimiento
    conteo_sentimientos = reseñas_filtradas['sentiment_category'].value_counts().to_dict()
    
    return conteo_sentimientos


@app.get("/sentiment_analysis/{year}")
async def read_sentiment_analysis(year: int):
    resultado = SentimentAnalysis(year)
    return {"resultado": resultado}


# MODELO ML: Función para recomendar juegos similares

def recomendacion_juego(product_id):
    # Especifica el tamaño de la muestra
    tamano_muestra = 1000
    
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

@app.get("/recommend/{product_id}", response_model=List[str])
def get_recommendations(product_id: int):
    recommendations = recomendacion_juego(product_id)
    return recommendations



