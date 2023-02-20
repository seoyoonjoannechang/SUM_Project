from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
[sys.path.append(i) for i in [".", "..", "Data/"]]

import pandas as pd
from Data.ModelData import predict_word

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
origins = ["*"]



@app.get("/")
async def predict():
   return predict_word()