# -*- coding: utf-8 -*-
import os
import logging
import json
from http import HTTPStatus

from dotenv import load_dotenv

# from fastapi import FastAPI
from fastapi import HTTPException, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from jurispacy_tokenizer import JuriSpacyTokenizer
from juritools.juriloss import JuriLoss
from juritools.predict import load_ner_model
import config

from utils import (
    get_juritools_info,
    process_ner,
)
from log_utils import (
    log_error,
    log_trace,
    log_on_startup,
    log_on_shutdown,
    add_custom_logger,
)
from pydantic import BaseModel
from juritools.type import NamedEntity, Decision


class NERResponse(BaseModel):
    entities: list[NamedEntity] = []
    checklist: list[str] = []


class JuritoolsInfo(BaseModel):
    version: str
    date: str


app = FastAPI(
    title="Pseudonymisation API",
    description="Predict named entities in French court decisions",
    version="1.0",
    disable_uvicorn_logger=True,
)

instrumentator = Instrumentator().instrument(app).expose(app)
app = add_custom_logger(
    app=app,
    custom_error_logger=log_error,
    custom_logger=log_trace,
)


@app.on_event("startup")
async def on_startup_events():
    log_on_startup()


@app.on_event("shutdown")
async def on_shutdown_events():
    log_on_shutdown()


@app.get("/")
def index():
    """Health check."""
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


@app.get("/juritools-info")
async def package_info():
    "Get version and date of juritools package"
    date, version = get_juritools_info()
    return JuritoolsInfo(version=version, date=date)


@app.post(
    "/ner",
    responses={
        200: {"description": "OK", "model": NERResponse},
        429: {"description": "API is busy"},
        422: {"description": "Data does not have the right shape"},
    },
)
def handler(decision: Decision):
    """Returns the tagged entities of the decision"""

    # Check if the endpoint is busy
    if config.processing_request:
        raise HTTPException(
            status_code=429,
            detail="Pseudonymisation in progress, endpoint is busy",
        )
    config.processing_request = True
    result = process_ner(decision, tokenizer, model)
    config.processing_request = False

    return NERResponse(**result)


@app.post("/loss")
async def loss(json_treatment: dict):
    """Returns the loss of a user treatment"""
    juriloss = JuriLoss(json_treatment, model=model, tokenizer=tokenizer)
    return juriloss.get_document_loss()


# Get the access to config file
load_dotenv()


# Windows Fix for PosixPath issue
if os.name == "nt":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

if os.environ.get("MODEL_JURICA"):
    model = load_ner_model(
        os.environ.get("MODEL_JURICA", "models/new_categories_model.pt"),
    )

else:
    logging.info(
        json.dumps(
            {
                "error_message": "No model found from ENV, please train a model first",
            },
            indent=4,
        )
    )
    raise EnvironmentError("MODEL_JURICA is not set")
# Load the tokenizer
tokenizer = JuriSpacyTokenizer()
