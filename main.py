from fastapi import FastAPI, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get("/developer", tags=["desarrollador"])

async def developer(developer):
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
    # Eliminamos la columna free.
    df.drop(columns=['free'], inplace=True)
    # Convertimos el dataframe a un diccionario.
    df = df.to_dict('records')
    # Devolvemos el resultado.
    return df

@app.get("/game_recommendation", tags=["recomendación"])

async def game_recommendation(item_id : str = Query(default='30')) :
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