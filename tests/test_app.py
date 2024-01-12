import datetime
import os
import re
import asyncio

import pytest
from httpx import Client, AsyncClient
from dotenv import load_dotenv


# from flair.models import SequenceTagger

# Windows Fix for PosixPath issue
if os.name == "nt":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

load_dotenv()

if API_URL := os.environ.get("API_URL"):
    client = Client(base_url=API_URL)
    async_client = AsyncClient(base_url=API_URL)
else:
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    async_client = AsyncClient(app=app, base_url="http://test")


pytest_plugins = "pytest_asyncio"


@pytest.mark.asyncio
async def test_multiple_call_ner_429():
    async with async_client as ac:
        call1 = ac.post(
            "/ner",
            json={
                "idLabel": "64f5aff6c9bbeeb075448279",
                "idDecision": "64f5b01596dfe49c47573aca",
                "sourceId": 2301729,
                "sourceName": "jurica",
                "text": "Pierre Dupont est ingénieur. "
                "Il est content de vivre à Nantes. "
                "Il travaille au tribunal de Paris. "
                "Il habite au 64 rue de Strasbourg 92400 Courbevoie. "
                "Il est hospitalisé à la clinique des Anges. "
                "Son compte bancaire es le 44437543.",
            },
        )
        call2 = ac.post(
            "/ner",
            json={
                "idLabel": "64f5aff6c9bbeeb075448279",
                "idDecision": "64f5b01596dfe49c47573aca",
                "sourceId": 2301729,
                "sourceName": "jurica",
                "text": "Pierre Dupont est ingénieur. "
                "Il est content de vivre à Nantes. "
                "Il travaille au tribunal de Paris. "
                "Il habite au 64 rue de Strasbourg 92400 Courbevoie. "
                "Il est hospitalisé à la clinique des Anges. "
                "Son compte bancaire es le 44437543.",
            },
        )

        res1, res2 = await asyncio.gather(call1, call2)

    assert res1.status_code == 200
    assert res2.status_code == 429


def test_ner_bad_meta_formatting():
    """Testing `/ner` endpoint with bad content"""
    response = client.post(
        "/ner",
        json={
            "idLabel": "64f5aff6c9bbeeb075448279",
            "idDecision": "64f5b01596dfe49c47573aca",
            "sourceId": 2301729,
            "sourceName": "jurica",
            "text": "blablabla",
            "parties": "",
        },
    )

    assert response.status_code == 422, response.content


def test_ner_simple():
    """Testing `/ner` endpoint with good content"""
    response = client.post(
        "/ner",
        json={
            "idLabel": "64f5aff6c9bbeeb075448279",
            "idDecision": "64f5b01596dfe49c47573aca",
            "sourceId": 2301729,
            "sourceName": "jurica",
            "text": "Pierre Dupont est ingénieur. "
            "Il est content de vivre à Nantes. "
            "Il travaille au tribunal de Paris. "
            "Il habite au 64 rue de Strasbourg 92400 Courbevoie. "
            "Il est hospitalisé à la clinique des Anges. "
            "Son compte bancaire est le 44437543.",
            "categories": ["personnePhysique", "localite", "adresse", "personneMorale"],
        },
    )

    assert response.status_code == 200, response.content

    results = response.json()
    scores = [entity["score"] for entity in results["entities"]]

    expected_results = {
        "entities": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
                "entityId": "personnePhysique_pierre",
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
                "entityId": "personnePhysique_dupont",
            },
            {
                "text": "Nantes",
                "start": 55,
                "end": 61,
                "label": "localite",
                "source": "postprocess",
                "score": scores[2],
                "entityId": "localite_nantes",
            },
            {
                "text": "64 rue de Strasbourg 92400 Courbevoie",
                "start": 111,
                "end": 148,
                "label": "adresse",
                "source": "NER model",
                "score": scores[3],
                "entityId": "adresse_64 rue de strasbourg 92400 courbevoie",
            },
            {
                "text": "clinique des Anges",
                "start": 174,
                "end": 192,
                "label": "personneMorale",
                "source": "NER model",
                "score": scores[4],
                "entityId": "personneMorale_clinique des anges",
            },
        ],
        "checklist": [
            "Il semblerait qu'un ou plusieurs numéros de comptes bancaires n'aient pas été repérés."  # noqa: E501
        ],
    }

    assert results.get("checklist") == expected_results.get("checklist")
    assert results.get("entities") == expected_results.get("entities")


def test_ner_with_categories():
    """Testing `/ner` endpoint with good content and categories"""
    response = client.post(
        "/ner",
        json={
            "idLabel": "64f5aff6c9bbeeb075448279",
            "idDecision": "64f5b01596dfe49c47573aca",
            "sourceId": 2301729,
            "sourceName": "jurica",
            "text": "Pierre Dupont est ingénieur. Il habite au 77 boulevard Saint-Germain à Paris",  # noqa: E501
            "categories": ["personnePhysique"],
        },
    )
    results = response.json()
    scores = [entity["score"] for entity in results["entities"]]
    assert results == {
        "entities": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
                "entityId": "personnePhysique_pierre",
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
                "entityId": "personnePhysique_dupont",
            },
        ],
        "checklist": [],
    }


def test_empty_text():
    """Testing `/ner` content with empty text"""
    response = client.post(
        "/ner",
        json={
            "idLabel": "64f5aff6c9bbeeb075448279",
            "idDecision": "64f5b01596dfe49c47573aca",
            "sourceId": 2301729,
            "sourceName": "jurica",
            "text": "",
        },
    )

    assert response.status_code == 422, response.content

    assert response.json() == {
        "detail": [
            {
                "type": "value_error",
                "loc": ["body", "text"],
                "msg": "Value error, text field is empty",
                "input": "",
                "ctx": {"error": {}},
                "url": "https://errors.pydantic.dev/2.3/v/value_error",
            }
        ]
    }


def test_get_juritools_info():
    response = client.get("/juritools-info")
    version_re = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")

    assert response.status_code == 200

    data = response.json()

    assert datetime.datetime.fromisoformat(data["date"]), data
    assert version_re.match(string=data["version"]), data


def test_ner_with_categories_tj():
    """Testing `/ner` endpoint with good content and categories"""
    response = client.post(
        "/ner",
        json={
            "idLabel": "64f5aff6c9bbeeb075448279",
            "idDecision": "64f5b01596dfe49c47573aca",
            "sourceId": 2301729,
            "sourceName": "juritj",
            "text": "Pierre Dupont est ingénieur. Il habite au 77 boulevard Saint-Germain à Paris",  # noqa: E501
            "categories": ["personnePhysique"],
        },
    )
    results = response.json()
    scores = [entity["score"] for entity in results["entities"]]
    assert results == {
        "entities": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
                "entityId": "personnePhysique_pierre",
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
                "entityId": "personnePhysique_dupont",
            },
        ],
        "checklist": [],
    }
