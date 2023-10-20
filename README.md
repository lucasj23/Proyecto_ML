![26TZ62LNNNHFLM5YYVOXEHM7HQ](https://github.com/lucasj23/Proyecto_ML/assets/131183621/43b0b67d-eb84-4eb3-9c0c-e17919c1b83f)


# Proyecto de Recomendación de Videojuegos - Machine Learning Operations (MLOps)

Nos encontramos ubicados hipoteticamente en el rol de un MLOps Engineer en un desafío emocionante, relacionado con la creación de un sistema de recomendación de videojuegos en Steam, una plataforma multinacional de videojuegos.

## Descripción del Problema y Rol a Desarrollar

### Contexto
Tenemos ya un modelo de recomendación de videojuegos que muestra buenas métricas, pero ahora se plantea el desafío de llevarlo al mundo real. El ciclo de vida de un proyecto de Machine Learning debe abarcar desde la recolección y el tratamiento de datos hasta el entrenamiento y el mantenimiento del modelo a medida que llegan nuevos datos.

### Rol a Desarrollar
Estamos trabajando como Data Scientist en Steam, y nos solicitaron la tarea de crear un sistema de recomendación de videojuegos para usuarios. Sin embargo, existen datos que presentan desafíos significativos, como datos anidados y falta de procesos automatizados para la actualización de productos. La misión es abordar estos desafíos y desarrollar un MVP (Producto Mínimo Viable) para resolver el problema.

## Propuesta de Trabajo (Requerimientos de Aprobación)

### Transformaciones de Datos
En este MVP, no se requieren necesariamente transformaciones de datos específicas, aunque se permite realizarlas si se identifica una justificación adecuada. Se permite eliminar columnas innecesarias para optimizar el rendimiento de la API y el entrenamiento del modelo.

### Feature Engineering
Tambien nos solicitaron crear la columna 'sentiment_analysis' en el dataset 'user_reviews' aplicando análisis de sentimiento con NLP (Natural Language Processing). Esta columna debe tomar el valor '0' si la reseña es mala, '1' si es neutral y '2' si es positiva. Si no es posible realizar el análisis por falta de reseñas escritas, debe tomar el valor de '1'.

### Desarrollo API
Se propuso disponibilizar los datos de la empresa usando el framework FastAPI. Las consultas propuestas son las siguientes:

- `PlayTimeGenre(genero: str)`: Devuelve el año con más horas jugadas para un género dado.
- `UserForGenre(genero: str)`: Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
- `UsersRecommend(año: int)`: Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
- `UsersNotRecommend(año: int)`: Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
- `sentiment_analysis(año: int)`: Según el año de lanzamiento, devuelve una lista con la cantidad de registros de reseñas de usuarios categorizados con un análisis de sentimiento.

### Deployment
Tenemos que considerar opciones de despliegue como Render, Railway u otros servicios que permitan que la API sea consumida desde la web. En nuestro caso, utilizaremos Render por cuestiones de practicidad.

### Análisis Exploratorio de los Datos (EDA)
Una vez que los datos estén limpios, será momento de investigar las relaciones entre las variables del dataset, identificar outliers o anomalías, y buscar patrones interesantes que puedan ser útiles en análisis posteriores. Nos recomendaron que evitemos el uso de librerías automáticas para EDA para poner en práctica los conceptos y tareas involucrados de forma manual.

### Modelo de Aprendizaje Automático
Después de que los datos sean accesibles a través de la API y se realice un análisis exploratorio completo, tendremos que entrenar un modelo de Machine Learning para crear un sistema de recomendación. Nos dieron para elegir entre un sistema de recomendación item-item (juego-juego) o user-item (usuario-juego) y proporcionar una función en la API para recibir recomendaciones.
En nuestro caso utilizamos un Modelo de Recomendacion item-item (es decir, juego - juego)

### Video de Presentación
Además de los mencionado, nos solicitaron crear un video de no más de 7 minutos donde se muestre el funcionamiento de la API y se explique brevemente el modelo utilizado para el sistema de recomendación. Lo esencial del video es la presentación y contextualizar el proyecto al inicio del video.

## A modo de comentario, nos enviaron los criterios a tener en cuenta en este proyecto: 

- Prolijidad del código.
- Uso de clases y/o funciones, en caso de ser necesario.
- Código comentado y legible.
- Estructura organizada del repositorio.
- README.md que presente el proyecto y el trabajo realizado de manera comprensible.

## Fuente de datos:
- [Dataset](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj): Carpeta con el archivo que requieren ser procesados, tengan en cuenta que hay datos que estan anidados (un diccionario o una lista como valores en la fila).
- [Diccionario de datos](https://docs.google.com/spreadsheets/d/1-t9HLzLHIGXvliq56UE_gMaWBVTPfrlTf2D9uAtLGrk/edit#gid=0): Diccionario con algunas descripciones de las columnas disponibles en el dataset.
