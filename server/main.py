from fastapi import FastAPI
from predict_trattamento import router as treatment_router
from predict_etdrs import router as etdrs_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eye-deseas.vercel.app"],  # <- URL esatto del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(treatment_router, prefix="/predict-treatment")
app.include_router(etdrs_router, prefix="/predict-etdrs")
