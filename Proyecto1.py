#!/usr/bin/env python
# coding: utf-8

# # Proyecto Individual 1 - ML OPS

# La empresa en donde estoy trabajando como Data Scientist, una una plataforma multinacional de videojuegos, me ha solicitado que me encargue de crear un sistema de recomendación de videojuegos para usuarios. La idea es crear un modelo de ML que solucione este problema de negocio. 
# 
# Dado que la madurez de los datos es nula, es necesario empezar desde 0, empezando por hacer un trabajo breve de Data Engineer y luego lograr tener un MVP (Minimum Viable Product) para el cierre del proyecto:
# 
# - Transformaciones: Para este MVP no se solicitan transformaciones de datos (aunque haya motivos para hacerlo) pero se trabajará en leer el dataset con el formato correcto. Me indicaron que se puede eliminar las columnas que no se necesitan para responder las consultas o preparar los modelos de aprendizaje automático, y de esa manera optimizar el rendimiento de la API y el entrenamiento del modelo.
# 
# - Feature Engineering: En el dataset 'user_reviews' se incluyen reseñas de juegos hechos por distintos usuarios. Se debe crear la columna 'sentiment_analysis' aplicando análisis de sentimiento con NLP con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta nueva columna debe reemplazar la de user_reviews.review para facilitar el trabajo de los modelos de machine learning y el análisis de datos. De no ser posible este análisis por estar ausente la reseña escrita, se indicó que debe tomar el valor de 1.
# 
# - Desarrollo API: Se propone disponibilizar los datos de la empresa usando el framework 'FastAPI'.

# Empezamos, entonces, con lo primero: importar nuestros datasets otorgados (los cuales son 3), para poder ver con que nos enfrentamos y poder realizar los cambios pertinentes.

# # # Importamos el primer dataset, 'Australian User Reviews' y realizamos ciertas modificaciones para que se pueda trabajar con el dataframe:

# In[1]:


# Importamos primer dataset: Australian Users Reviews

import pandas as pd
import ast

# Lista para almacenar los diccionarios JSON de cada línea
data_list = []

# Ruta del archivo JSON
file_path = 'australian_user_reviews.json'

# Abrir el archivo y procesar cada línea
with open(file_path, 'r', encoding='utf8') as file:
    for line in file:
        try:
            # Usar ast.literal_eval para convertir la línea en un diccionario
            json_data = ast.literal_eval(line)
            data_list.append(json_data)
        except ValueError as e:
            print(f"Error en la línea: {line}")
            continue

# Crear un DataFrame a partir de la lista de diccionarios
df_reviews = pd.json_normalize(data_list, record_path='reviews', meta=['user_id','user_url'])


# In[2]:


# Ver los primeros registros del DataFrame
df_reviews.head()


# In[3]:


# verificamos la informacion rapidamente: 
df_reviews.info()


# In[4]:


df_reviews = df_reviews.dropna(thresh=2)
df_reviews.info()


# In[5]:


# Modificamos tipo de dato:

df_reviews['item_id'] = df_reviews['item_id'].astype(int)


# In[6]:


df_reviews.info()


# In[7]:


# Eliminamos las columnas que no nos serviran a los fines de nuestro análisis: 

df_reviews.drop(columns=['funny', 'last_edited', 'helpful'], inplace=True)


# Procedemos a modificar el formato de la columna 'posted' para luego poder trabajar con ella:

# In[8]:


from dateutil import parser 

# Funcion para analizar la fecha y manejar errores: 
def parse_date(date_str): 
    try:
        return parser.parse(date_str.replace("Posted ", ""), fuzzy = True)
    except ValueError:
        return None
    
# Aplicamos la funcion de analisis de fecha y reemplazamos las filas que tienen fechas incorrectas con NaN
df_reviews['posted'] = df_reviews['posted'].apply(parse_date)

# Además eliminamos esas filas que contengan valores NaN (ya que son fechas incorrectas y no nos servirán posteriormente en el análisis):
df_reviews = df_reviews.dropna(subset=['posted'])


# In[10]:


df_reviews['posted'] = pd.to_datetime(df_reviews['posted'])


# In[11]:


df_reviews.info()


# # Importamos el segundo dataset, 'Australian User Items' y realizamos ciertas modificaciones para que se pueda trabajar con el dataframe:
# 

# In[12]:


data_list2 = []

# Ruta del archivo JSON
file_path2 = 'australian_users_items.json'

# Abrir el archivo y procesar cada línea
with open(file_path2, 'r', encoding='utf8') as file:
    for line in file:
        try:
            # Usar ast.literal_eval para convertir la línea en un diccionario
            json_data2 = ast.literal_eval(line)
            data_list2.append(json_data2)
        except ValueError as e:
            print(f"Error en la línea: {line}")
            continue

# Crear un DataFrame a partir de la lista de diccionarios
df_items = pd.json_normalize(data_list2, record_path='items', meta='user_id')


# In[13]:


# Ver los primeros registros del DataFrame
df_items.head()


# In[14]:


# Dropeamos las columnas innecesarias
df_items.drop(columns=['playtime_2weeks'], inplace=True) 


# In[15]:


df_items['item_id'] = df_items['item_id'].astype('int32')


# In[16]:


# Eliminamos duplicados: 
df_items = df_items.drop_duplicates()


# In[17]:


df_items.info()


# In[18]:


df_items.isnull().sum()


# In[19]:


columnas_a_convertir = ['user_id', 'item_id', 'playtime_forever']
# Iterar a través de las columnas y convertirlas a int, manejando los errores
for columna in columnas_a_convertir:
    df_items[columna] = pd.to_numeric(df_items[columna], errors='coerce')


# In[20]:


df_items.info()


# In[21]:


df_items.isnull().sum()


# In[22]:


df_items = df_items.dropna()


# In[23]:


df_items.isnull().sum()


# In[24]:


df_items.info()


# In[25]:


# filas_con_none = df_items.isna().sum(axis=1)

# Filtrar el DataFrame original para obtener filas con más de 3 'None'
# filas_mas_de_3_none = df_items[filas_con_none > 3]
# filas_mas_de_3_none
# ejecutamos esto y vemos que no hay registros que tengan mas de 3 columnas con None. 


# # Importamos tercer dataset, 'Output Steam Games' y realizamos ciertas modificaciones para que se pueda trabajar con el dataframe: 
# 

# In[26]:


df_games = pd.read_json('output_steam_games.json', lines=True)


# In[27]:


# Ver los primeros registros del DataFrame
df_games.head()


# In[28]:


df_games = df_games.dropna(thresh=5)
df_games.head()


# In[29]:


df_games.columns


# In[30]:


# Borramos las columnas que no nos servirán a los fines de nuestor análisis: 

df_games.drop(columns=['reviews_url', 'specs', 'early_access','app_name','publisher'], inplace=True) 


# In[31]:


df_games.info()


# In[32]:


# Procedemos a modificar el formato de ciertas columnas:
# En primer lugar, las fechas de la columna 'release date' para luego poder trabajar con ella:

df_games['release_date'] = pd.to_datetime(df_games['release_date'], format='%Y-%m-%d', errors='coerce')


# In[33]:


# Convertimos en 0 los registros que no tengan ID: 
df_games['id'] = df_games['id'].fillna(0).astype(int)


# In[34]:


# Ahora convertimos todos los ID en enteros, para poder trabajar mejor: 
df_games['id'] = df_games['id'].astype(int)


# In[35]:


df_games.head()


# Sobre las reseñas de juegos hechos por distintos usuarios, las cuales se encuentran en el dataset 'user_reviews', procedemos a aplicar el Análisis de sentimiento con NLP. 
# Como fue ordenado, se debe tomar el valor '0' si el review es malo, '1' si es neutral y '2' si es positivo. En el caso de no ser posible este análisis por estar ausente la reseña escrita, debe tomar el valor de 1.

# In[36]:


from textblob import TextBlob

# Función para realizar el análisis de sentimiento y asignar valores
def analyze_sentiment(text):
    if pd.isna(text):  # Verificar si el texto está ausente
        return 1  # Fijamos en 1 el valor que devolverá si el texto está ausente
    else:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        # utilizamos un umbral de +- 0.2 para tener una clasificación más refinada
        # y distinguir entre reseñas ligeramente positivas o negativas y las que son claramente positivas o negativas: 
        if sentiment < -0.2:  # Valor negativo, consideramos que es 'malo'
            return 0
        elif sentiment > 0.2:  # Valor positivo, consideramos que es 'positivo'
            return 2
        else:  # Valor neutral
            return 1

# Aplicar la función a la columna 'review' y reemplazarla con una nueva columna de nombre 'sentiment_analysis'
df_reviews['sentiment_analysis'] = df_reviews['review'].apply(analyze_sentiment)

# Mostrar el DataFrame resultante
print(df_reviews)


# In[37]:


df_reviews.info()


# In[38]:


# Verificamos que hayan sido puntuadas todas las filas: 

count_0 = (df_reviews['sentiment_analysis'] == 0).sum()
count_1 = (df_reviews['sentiment_analysis'] == 1).sum()
count_2 = (df_reviews['sentiment_analysis'] == 2).sum()

print(f"Número de filas con valor 0: {count_0}")
print(f"Número de filas con valor 1: {count_1}")
print(f"Número de filas con valor 2: {count_2}")
print(f"Total puntuadas: {count_0 + count_1 + count_2}")


# # Exportamos los dataframes ya modificados (es decir listos para poder trabajar nuestras funciones) a formato CSV y PARQUET y los volvemos a leer, para trabajar de forma mas prolija: 

# Cabe aclarar que exportamos dos dataframes a formato CSV y uno de ellos a formato PARQUET por una cuestion de tamaño. El tamaño del dataframe exportado a PARQUET era más grande que los demas, imposibilitando su trabajo en formato CSV y por eso se elije formato PARQUET. 
# 
# Les asignamos el mismo nombre que antes para no confundirnos.

# In[39]:


# Exportar df_reviews a un archivo CSV
df_reviews.to_csv('df_reviews.csv', index=False)  # El parámetro index=False evita que se incluya el índice en el archivo CSV

# Exportar df_games a un archivo CSV
df_games.to_csv('df_games.csv', index=False)

# Exportar df_items a un archivo Parquet
df_items.to_parquet('df_items.parquet', index=False)  # El parámetro index=False evita que se incluya el índice en el archivo Parquet


# In[40]:


# Leemos los archivos: 
import pyarrow.parquet as pq

df_csv1 = pd.read_csv('df_reviews.csv')
df_csv2 = pd.read_csv('df_games.csv')
df_parquet = pq.read_table('df_items.parquet').to_pandas()


# # Funcion Nro. 1:
# def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.

# In[41]:


# Primero definimos una función para convertir un objeto en una cadena de texto:
def ensure_string2(obj):
    if isinstance(obj, str):
        # Intenta evaluar la cadena como una lista
        try:
            obj_list = ast.literal_eval(obj)
            if isinstance(obj_list, list):
                return ', '.join(obj_list)
        except ValueError:
            pass
    # Si no podemos evaluarlo como una lista, simplemente lo dejamos como está
    return str(obj)


# In[42]:


# aplicamos la funcion:
df_games['genres'] = df_games['genres'].apply(ensure_string2)


# In[43]:


# Ahora si redactamos la funcion solicitada: 
def PlayTimeGenre(genero, df_games, df_items):
    games_filtered = df_games[df_games['genres'].str.contains(genero, case=False, na=False)]

    # Filter df_items to obtain rows with similar item_id
    merged_df = pd.merge(df_items, games_filtered, left_on='item_id', right_on='id', how='inner')
    merged_df['release_date'] = pd.to_datetime(merged_df['release_date'])
    merged_df['release_year'] = merged_df['release_date'].dt.year

    grouped = merged_df.groupby('release_year')['playtime_forever'].sum()

    max_year = grouped.idxmax()

    return f"Año con más horas jugadas para Género {genero}: {int(max_year)}"


# In[44]:


resultado = PlayTimeGenre('Adventure', df_csv2, df_parquet)
print(resultado)


# # Función Nro. 2: 
#  def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

# In[45]:


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



# In[46]:


resultado2 = UserForGenre('Sports', df_csv2, df_parquet)
print(resultado2)


# # Funcion Nro. 3: 
# "def UsersRecommend( año : int )": Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)

# Para poder realizar la funcion, acorde a lo solicitado, previamente tenemos que realizar un merge fuera de la función y guarda el resultado en un DataFrame: 
# 

# In[47]:


merged_df2 = pd.merge(df_reviews, df_games, left_on='item_id', right_on='id', how='inner')
merged_df2.head()


# In[48]:


merged_df2 = merged_df2[['posted', 'title', 'sentiment_analysis', 'recommend', 'release_date']]


# In[49]:


merged_df2.to_csv('merged_df2.csv')


# In[69]:


merged_df2.head()


# In[71]:


# Ahora procedemos a desarrollar la funcion: 
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


# In[72]:


# Realizamos la verificacion de que funcione: 

resultado3 = UsersRecommend(2012)
print(resultado3)


# # Funcion Nro. 4: 
# "def UsersNotRecommend( año : int )": Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)

# In[54]:


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



# In[56]:


# Realizamos la verificacion de que funcione: 

resultado4 = UsersNotRecommend(2011)
print(resultado4)


# In[57]:


# Aca, por otro lado, y para verificar completamente la funcion, verificamos si realmente lo que nos está devolviendo coincide con los criterios de sentiment analysis que
# realizamos previamente, y que justamente es en lo que se basa la funcion anterior: 
def VerificarReseñas(año: int):
    global df_reviews
    df_reviews['posted'] = pd.to_datetime(df_reviews['posted'])
    # Filtra las reseñas para el año dado y que tengan no recomendación (recommend = False) y comentarios negativos (sentiment_analysis < 0)
    reseñas_filtradas = df_reviews[(df_reviews['posted'].dt.year == año) &
                                  (df_reviews['recommend'] == False) & (df_reviews['sentiment_analysis'] == 0)]

    # Verifica si hay reseñas que cumplen con los criterios
    if not reseñas_filtradas.empty:
        print("Hay reseñas para el año {} que cumplen con los criterios.".format(año))
    else:
        print("No hay reseñas para el año {} que cumplan con los criterios.".format(año))

# Llamada a la función para verificar las reseñas para un año específico
VerificarReseñas(2011)  # Cambia el año según tus datos


# # Funcion Nro. 5: 
# 'def sentiment_analysis( año : int )': Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

# In[61]:


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




# In[62]:


# Ejemplo de uso
resultado5 = SentimentAnalysis(2013)  # Cambia el año según tus datos
print(resultado5)


# In[ ]:




