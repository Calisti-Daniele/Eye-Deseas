from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import io

# Inizializza FastAPI
router = APIRouter()

# Carica il modello una sola volta
with open('models/catboost_model_no_visits.pkl', 'rb') as f:
    model = pickle.load(f)

@router.post("/")
async def predict(file: UploadFile = File(...)):
    # Legge il file in memoria
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=";")

    # Estrai solo le colonne numeriche
    X = df.select_dtypes(include=['number'])

    # Debug (facoltativo)
    print("Colonne nel file:", df.columns.tolist())
    print("Colonne numeriche:", X.columns.tolist())

    print("DataFrame intero:")
    print(df.head())

    print("Colonne numeriche:")
    print(X.head())

    print("Shape X:", X.shape)

    # Predizione
    prediction = model.predict(X)

    # Restituisce la predizione
    return {
        "prediction": prediction.tolist()[0]  # visto che è una sola riga
    }
# 2 monthly 0 gila 1 trex
