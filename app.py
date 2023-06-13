# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from jurispacy_tokenizer import JuriSpacyTokenizer
from juritools.juriloss import JuriLoss
from juritools.predict import load_ner_model

from models import Decision, NERResponse
from utils import get_juritools_info
from utils import ner

# Windows Fix for PosixPath issue
if os.name == "nt":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI(
    title="Pseudonymisation API",
    description="Predict named entities in French court decisions",
    version="1.0",
)


@app.get("/")
def _index():
    """Health check."""
    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {}}


@app.get("/juritools-info")
async def package_info():
    "Get version and date of juritools package"
    date, version = get_juritools_info()
    return {"version": version, "date": date}


@app.post("/ner",
          responses={
                200: {
                    "description": "OK",
                    "model": NERResponse
                },
                429: {
                    "description": "API is busy"
                },
                422: {
                    "description": "Data does not have the right shape"
                }
})
def handler(decision: Decision):
    """Returns the tagged entities of the decision"""
    
    # Check if the endpoint is busy
    global processing_request
    if processing_request:
        raise HTTPException(
            status_code=429,
            detail="Pseudonymisation in progress, endpoint is busy",
        )
    processing_request = True
    with ThreadPoolExecutor() as executor:
        future = executor.submit(ner, decision, tokenizer, model)
        result = future.result()

    processing_request = False

    return result


@app.post("/loss")
async def loss(json_treatment: Dict):
    """Returns the loss of a user treatment"""
    juriloss = JuriLoss(json_treatment, model=model, tokenizer=tokenizer)
    return juriloss.get_document_loss()


# Get the access to config file
load_dotenv()

if os.getenv("MODEL"):
    model = load_ner_model(
        os.path.join(
            os.path.dirname(__file__),
            os.environ.get("MODEL", "model/sequence_tagger.pt"),
        )
    )
    print("Model loaded")
else:
    print("No model found from ENV, please train a model first")

# Load the tokenizer
tokenizer = JuriSpacyTokenizer()
print("Tokenizer loaded")

processing_request = False
