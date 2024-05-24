import fastapi
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = fastapi.FastAPI()

@app.get("/developer", tags=["desarrollador"])

async def developer(developer):

    df = pd.read_parquet('./Datasets/developer_endpoint.parquet')
    df = df[df['developer'] == developer]
    df['Año'] = df['release_date'].dt.year
    df_year = df.groupby('Año').size().reset_index(name='Cantidad de Items')
    df_free = df.groupby('Año')['free'].sum().reset_index(name='free')
    df = pd.merge(df_year, df_free, on='Año')
    df['Contenido Free'] = round(df['free'] / df['Cantidad de Items'] * 100, 2)
    df.drop(columns=['free'], inplace=True)
    df = df.to_dict('records')
    return df

df = pd.read_parquet('./Datasets/game_recommendation.parquet')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.get("/game_recommendation", tags=["recomendación"])

async def game_recommendation(item_id):
    item_id = str(item_id)
    if item_id not in df['item_id'].values:
        return 'ID no encontrado'
    idx = df[df['item_id'] == item_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    recommended_games = df['title'].iloc[[i[0] for i in sim_scores]].tolist()
    return {"recommended_games": recommended_games}