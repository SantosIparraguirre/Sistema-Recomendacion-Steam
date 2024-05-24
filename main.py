import fastapi
import pandas as pd

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de consultas y recomendación de videojuegos de Steam"}

@app.get("/developer", tags=["desarrollador"])

async def developer(desarrollador):

    df = pd.read_parquet('./Datasets/developer_endpoint.parquet')
    df = df[df['developer'] == desarrollador]
    df['Año'] = df['release_date'].dt.year
    df_year = df.groupby('Año').size().reset_index(name='Cantidad de Items')
    df_free = df.groupby('Año')['free'].sum().reset_index(name='free')
    df = pd.merge(df_year, df_free, on='Año')
    df['Contenido Free'] = round(df['free'] / df['Cantidad de Items'] * 100, 2)
    df.drop(columns=['free'], inplace=True)
    df = df.to_dict('records')
    return df