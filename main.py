from fastapi import FastAPI, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get("/developer", tags=["Desarrollador"])

async def developer(developer : str = Query(default='Valve', description='Ingrese el nombre de un desarrollador. Ejemplo: Kotoshiro. Salida: Cantidad de items y porcentaje de contenido free por año')):
    # Cargamos el dataset.
    df = pd.read_parquet('./Datasets/developer_endpoint.parquet')

    # Filtramos por el desarrollador ingresado.
    df = df[df['developer'] == developer]

    # Creamos la columna año en base a la columna release_date.
    df['Año'] = df['release_date'].dt.year

    # Agrupamos por año y contamos la cantidad de items.
    df_year = df.groupby('Año').size().reset_index(name='Cantidad de Items')

    # Agrupamos por año y sumamos la cantidad de items free.
    df_free = df.groupby('Año')['free'].sum().reset_index(name='free')

    # Hacemos un merge de ambos dataframes mediante la columna Año.
    df = pd.merge(df_year, df_free, on='Año')

    # Calculamos el porcentaje de contenido free.
    df['Contenido Free'] = round(df['free'] / df['Cantidad de Items'] * 100, 2)

    # Convertimos el porcentaje a string y le agregamos el símbolo %.
    df['Contenido Free'] = df['Contenido Free'].apply(lambda x: f'{x}%')

    # Eliminamos la columna free.
    df.drop(columns=['free'], inplace=True)

    # Convertimos el dataframe a un diccionario.
    df = df.to_dict('records')

    # Devolvemos el resultado.
    return df

@app.get("/user_data", tags=["Datos del usuario"])

async def user_data(user_id : str = Query(default='mayshowganmore', description='Ingrese el ID de un usuario. Ejemplo: 76561197970982479. Salida: Dinero gastado, porcentaje de recomendación y cantidad de items')):
    # Cargamos los datasets.
    df_user_reviews = pd.read_parquet('./Datasets/user_reviews_preprocessed.parquet')
    df_games = pd.read_parquet('./Datasets/steam_games_preprocessed.parquet')
    df_user_items = pd.read_parquet('./Datasets/users_items_preprocessed.parquet')

    # Convertimos user_id a string y user_id de df_user_items a string.
    user_id = str(user_id)
    df_user_items['user_id'] = df_user_items['user_id'].astype(str)

    # Filtramos df_user_items por user_id.
    df_user_items = df_user_items[df_user_items['user_id'] == user_id]

    # Obtenemos los títulos de los juegos del usuario y sus precios.
    user_game_titles = df_user_items['item_name'].tolist()
    user_game_prices = df_games[df_games['title'].isin(user_game_titles)]['price'].tolist()

    # Calculamos el dinero total gastado por el usuario.
    total_money_spent = round(sum(user_game_prices), 2)

    # Filtramos df_user_reviews por user_id.
    user_reviews = df_user_reviews[df_user_reviews['user_id'] == user_id]

    # Si no está vacío, calculamos el porcentaje de recomendación.
    if not user_reviews.empty:
        recommend_count = user_reviews['recommend'].value_counts(normalize=True)
        recommend_percentage = round(recommend_count.get(True, 0) * 100, 2)

    # Si está vacío, asignamos 0 al porcentaje de recomendación.
    else:
        recommend_percentage = 0

    # Obtenemos la cantidad de items del usuario.
    items_count = df_user_items[df_user_items['user_id'] == user_id].shape[0]

    # Devolvemos el resultado.
    return {
        'Usuario': user_id,
        'Dinero gastado': f'{total_money_spent} USD',
        '% de recomendación': f'{recommend_percentage}%',
        'Cantidad de items': items_count
    }

@app.get("/user_for_genre", tags=["Usuario con más horas jugadas para un género"])

async def user_for_genre(genre: str = Query(default='Action', description='Ingrese un género. Ejemplo: RPG. Salida: Usuario con más horas jugadas para el género ingresado y cantidad de horas jugadas por año')):
    # Cargamos el dataset.
    df = pd.read_parquet('./Datasets/userforgenre_endpoint.parquet')
    
    # Filtramos por el género ingresado.
    df = df[df['genres'].str.contains(genre, case=False, na=False)]

    if df.empty:
        return {'Género no encontrado'}
    
    # Guardamos en la variable user_hours la suma de las horas jugadas por usuario.
    user_hours = df.groupby('user_id')['playtime_forever'].sum().reset_index()
    
    # Obtenemos el usuario con más horas jugadas y guardamos su ID en la variable top_user_id.
    top_user = user_hours.loc[user_hours['playtime_forever'].idxmax()]
    top_user_id = top_user['user_id']
    
    # Guardamos en la variable top_user_df las filas del dataframe que corresponden al usuario con más horas jugadas.
    top_user_df = df[df['user_id'] == top_user_id]
    
    # Creamos la columna year en base a la columna release_date.
    top_user_df['year'] = top_user_df['release_date'].dt.year

    # Agrupamos por año y sumamos las horas jugadas.
    hours_by_year = top_user_df.groupby('year')['playtime_forever'].sum().reset_index()
    
    # Renombramos 'year' por 'Año' y 'playtime_forever' por 'Horas jugadas'.
    hours_by_year.rename(columns={'year': 'Año', 'playtime_forever': 'Horas jugadas'}, inplace=True)

    # Convertimos hours_by_year en diccionario.
    hours_by_year_dict = hours_by_year.to_dict(orient='records')
    
    # Devolvemos el resultado.
    return {
        f"Usuario con más horas jugadas para el género {genre}": top_user_id,
        "Horas jugadas por año": hours_by_year_dict
    }

@app.get("/best_developer_year", tags=["Top 3 desarrolladores por año"])

async def best_developer_year(year: int = Query(default=2000, description='Ingrese un año. Ejemplo: 2005. Salida: Top 3 desarrolladores con más juegos recomendados y reseñas positivas para el año ingresado.')):
    # Cargamos el dataset.
    df = pd.read_parquet('./Datasets/best_developer_year_endpoint.parquet')

    # Creamos la columna year en base a release_date.
    df['year'] = df['release_date'].dt.year

    # Filtramos por el año ingresado.
    df = df[df['year'] == year]

    # Si el año ingresado no coincide, devolvemos un mensaje de error.
    if df.empty:
        return {'Año no encontrado'}

    # Filtramos por recomendaciones y reseñas positivas.
    df = df[(df['recommend'] == True) & (df['sentiment_analysis'] == 2)]

    # Agrupamos por desarrollador y contamos la cantidad de recomendaciones.
    df = df.groupby('developer').size().reset_index(name='recommend_count')

    # Ordenamos de mayor a menor y obtenemos los 3 primeros.
    df = df.sort_values(by='recommend_count', ascending=False).head(3)

    # Formateamos el resultado como lista de diccionarios.
    df = [{"Puesto {}".format(i+1): row[1]} for i, row in enumerate(df.itertuples())]

    # Devolvemos el resultado.
    return df

@app.get("/game_recommendation", tags=["Recomendación de videojuegos"])

async def game_recommendation(item_id : str = Query(default='10', description='Debe ingresar un ID de juego. Ejemplo: 10 = Counter-Strike. Salida: Lista de 5 juegos recomendados basados en similitud de contenido.')):
    # Cargamos el dataset.
    df = pd.read_parquet('./Datasets/game_recommendation.parquet')

    # Verificamos si el item_id ingresado se encuentra en el dataset.
    if item_id not in df['item_id'].values:

    # Si no se encuentra, devolvemos un mensaje de error.
        return {'ID no encontrado'}
    
    # Creamos una instancia de TfidfVectorizer con las stopwords en inglés.
    tfidf = TfidfVectorizer(stop_words='english')

    # Creamos la matriz tf-idf de los features.
    tfidf_matrix = tfidf.fit_transform(df['features'])

    # Calculamos la similitud de coseno entre los items.
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Obtenemos el índice del item_id ingresado.
    idx = df[df['item_id'] == item_id].index[0]

    # Obtenemos los scores de similitud coseno.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenamos los scores de mayor a menor.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtenemos los 5 items más similares.
    sim_scores = sim_scores[1:6]

    # Obtenemos los títulos de los items recomendados y los convertimos en lista.
    recommended_games = df['title'].iloc[[i[0] for i in sim_scores]].tolist()

    # Devolvemos el resultado.
    return {"recommended_games": recommended_games}