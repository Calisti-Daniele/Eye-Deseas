from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from keras.api.models import load_model
import io
import traceback

router = APIRouter()

# Static features per trattamento
STATIC_FEATURES_MAP = {
    'TREX': ['age', 'gender', 'insulinuser', 'smoker', 'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi'],
    'MONTHLY': ['age_at_enrollment', 'gender', 'insulinuser', 'smoker', 'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi'],
    'GILA': ['age', 'gender', 'insulinuser', 'smoker', 'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi'],
}

def get_column_name(i, treatment):
    return f'etdrs_{i}_visit' if treatment == 'TREX' else f'etdrs_visit_{i}'

def prepare_input_sequence(current_visits, static_values, total_timesteps=19):
    n = len(current_visits)
    x_seq = np.array(current_visits, dtype=np.float32).reshape(-1, 1)
    static_rep = np.tile(static_values, (n, 1))
    x_combined = np.concatenate([x_seq, static_rep], axis=1)

    pad_len = total_timesteps - n
    if pad_len > 0:
        padding = np.zeros((pad_len, x_combined.shape[1]), dtype=np.float32)
        x_combined = np.vstack([x_combined, padding])

    return x_combined

def predict_sequence_gru(row, model, scaler_y, static_features, treatment, start_n=3, target_n=20):
    current_visits = [row[get_column_name(i, treatment)] for i in range(1, start_n + 1)]
    static = row[static_features].values.astype(np.float32)

    for t in range(start_n + 1, target_n + 1):
        X_seq = prepare_input_sequence(current_visits, static)
        X_input = np.expand_dims(X_seq, axis=0)

        y_pred_norm = model.predict(X_input, verbose=0)[0][0]
        y_pred = scaler_y.inverse_transform([[y_pred_norm]])[0][0]

        row[get_column_name(t, treatment)] = y_pred
        current_visits.append(y_pred)

    return row

@router.post("/")
async def predict_etdrs(
    file: UploadFile = File(...),
    treatment: str = Form(...),
    start_n: int = Form(3),
    target: int = Form(20)
):
    treatment = treatment.upper()
    if treatment not in STATIC_FEATURES_MAP:
        return {"error": f"Trattamento '{treatment}' non supportato."}

    if target <= start_n or target > 20:
        return {"error": f"Il parametro target deve essere maggiore di start_n e al massimo 20."}

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=";")
    df = df.apply(pd.to_numeric, errors='coerce')

    static_features = STATIC_FEATURES_MAP[treatment]

    # Percorsi ai modelli
    model_path = f"models/{treatment.lower()}_scheduled_10k_bidirectional.keras"
    scaler_path = f"scaler/scaler_gru_y_{treatment.lower()}_10k_bidirectional.pkl"

    try:
        print(f"📂 Caricamento modello da: {model_path}")
        model = load_model(model_path)
        print("✅ Modello caricato con successo")

        print(f"📂 Caricamento scaler da: {scaler_path}")
        scaler_y = joblib.load(scaler_path)
        print("✅ Scaler caricato con successo")
    except Exception as e:
        print("❌ Errore nel caricamento del modello o dello scaler:")
        traceback.print_exc()
        return {
            "error": "Errore nel caricamento modello o scaler.",
            "dettagli": traceback.format_exc()
        }

    try:
        records = []
        for _, row in df.iterrows():
            row = predict_sequence_gru(row.copy(), model, scaler_y, static_features, treatment, start_n=start_n,
                                       target_n=target)
            records.append(row)

        df_pred = pd.DataFrame(records)
    except Exception as e:
        print()
        return {"error": f"Errore durante la predizione: {str(e)}"}

    prediction_columns = [get_column_name(i, treatment) for i in range(start_n + 1, target + 1)]
    predictions = df_pred[prediction_columns].to_dict(orient='records')

    return {
        "predictions": predictions
    }
