import os

import pandas as pd
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from jurispacy_tokenizer import JuriSpacyTokenizer
from juritools.predict import load_ner_model

from app import app
from utils import process_metadata

# from flair.models import SequenceTagger

# Windows Fix for PosixPath issue
if os.name == "nt":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

load_dotenv()

model = load_ner_model(
    os.path.join(
        os.path.dirname(__file__),
        os.environ.get("MODEL", "model/sequence_tagger.pt"),
    )
)


tokenizer = JuriSpacyTokenizer()

client = TestClient(app)


def test_ner_bad_meta_formatting():
    """Testing `/ner` endpoint with bad content"""
    response = client.post(
        "/ner", json={"idDocument": 2301729, "text": "blablabla", "meta": ""}
    )
    assert response.status_code == 422, response.content


def test_ner_simple():
    """Testing `/ner` endpoint with good content"""
    response = client.post(
        "/ner",
        json={
            "idDocument": 2301729,
            "text": "Pierre Dupont est ingénieur. "
            "Il est content de vivre à Nantes. "
            "Il travaille au tribunal de Paris. "
            "Il habite au 64 rue de Strasbourg 92400 Courbevoie. "
            "Il est hospitalisé à la clinique des Anges. "
            "Son compte bancaire es le 44437543.",
        },
    )

    assert response.status_code == 200, response.content

    results = response.json()
    scores = [entity["score"] for entity in results["postProcess"]]

    expected_results = {
        "tagger": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
            },
            {
                "text": "64 rue de Strasbourg 92400 Courbevoie",
                "start": 111,
                "end": 148,
                "label": "adresse",
                "source": "NER model",
                "score": scores[3],
            },
            {
                "text": "clinique des Anges",
                "start": 174,
                "end": 192,
                "label": "personneMorale",
                "source": "NER model",
                "score": scores[4],
            },
        ],
        "postProcess": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
            },
            {
                "text": "Nantes",
                "start": 55,
                "end": 61,
                "label": "localite",
                "source": "postprocess",
                "score": scores[2],
            },
            {
                "text": "64 rue de Strasbourg 92400 Courbevoie",
                "start": 111,
                "end": 148,
                "label": "adresse",
                "source": "NER model",
                "score": scores[3],
            },
            {
                "text": "clinique des Anges",
                "start": 174,
                "end": 192,
                "label": "personneMorale",
                "source": "NER model",
                "score": scores[4],
            },
        ],
        "output": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
            },
            {
                "text": "Nantes",
                "start": 55,
                "end": 61,
                "label": "localite",
                "source": "postprocess",
                "score": scores[2],
            },
            {
                "text": "64 rue de Strasbourg 92400 Courbevoie",
                "start": 111,
                "end": 148,
                "label": "adresse",
                "source": "NER model",
                "score": scores[3],
            },
            {
                "text": "clinique des Anges",
                "start": 174,
                "end": 192,
                "label": "personneMorale",
                "source": "NER model",
                "score": scores[4],
            },
        ],
        "checklist": [
            "Il semblerait qu'un ou plusieurs numéros de "
            "comptes bancaires n'aient pas été repérées"
        ],
    }

    assert results.get("checklist") == expected_results.get("checklist")
    assert results.get("entities") == expected_results.get("entities")


def test_ner_with_categories():
    """Testing `/ner` endpoint with good content and categories"""
    response = client.post(
        "/ner",
        json={
            "idDocument": 2301729,
            "text": "Pierre Dupont est ingénieur. Il habite au 77 boulevard Saint-Germain à Paris",
            "categories": ["personnePhysique"],
        },
    )
    results = response.json()
    scores = [entity["score"] for entity in results["postProcess"]]
    print(results)
    assert results == {
        "tagger": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
            },
            {
                "text": "77 boulevard Saint-Germain à Paris",
                "start": 42,
                "end": 76,
                "label": "adresse",
                "source": "NER model",
                "score": scores[2],
            },
        ],
        "postProcess": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
            },
            {
                "text": "77 boulevard Saint-Germain à Paris",
                "start": 42,
                "end": 76,
                "label": "adresse",
                "source": "NER model",
                "score": scores[2],
            },
        ],
        "output": [
            {
                "text": "Pierre",
                "start": 0,
                "end": 6,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[0],
            },
            {
                "text": "Dupont",
                "start": 7,
                "end": 13,
                "label": "personnePhysique",
                "source": "NER model",
                "score": scores[1],
            },
        ],
        "checklist": [],
    }


def test_empty_text():
    """Testing `/ner` content with empty text"""
    response = client.post("/ner", json={"idDocument": 2301729, "text": ""})

    assert response.status_code == 422, response.content

    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "text"],
                "msg": "text field is empty",
                "type": "value_error",
            }
        ]
    }


def test_process_metadata_jurinet():
    """Testing preprocessing of Jurinet metadata"""
    meta = [
        [
            1725609,
            "AVOCAT",
            12272125,
            0,
            None,
            "SCP",
            "Pat Patrouille",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "52011060",
        ],
        [
            1725609,
            "AVOCAT",
            12272125,
            0,
            None,
            "SCP",
            "Boul et Bill",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "52011060",
        ],
    ]
    metadata = process_metadata(meta, "jurinet", model, tokenizer)

    assert isinstance(metadata, pd.DataFrame)

    assert (
        metadata.to_csv(index=False).replace("\r", "")
        == "ID_DOCUMENT,TYPE_PERSONNE,ID_PARTIE,NATURE_PARTIE,TYPE_PARTIE,ID_TITRE,NOM,"
        "PRENOM,NOM_MARITAL,AUTRE_PRENOM,ALIAS,SIGLE,DOMICILIATION,LIG_ADR1,LIG_ADR2,LIG"
        "_ADR3,CODE_POSTAL,NOM_COMMUNE,NUMERO\n1725609,AVOCAT,12272125,0,,SCP,Pat Patro"
        "uille,,,,,,,,,,,,52011060\n1725609,AVOCAT,12272125,0,,SCP,Boul et Bill,,,,,,,,"
        ",,,,52011060\n"
    )


def test_process_metadata_jurica():
    """Testing preprocessing of Jurica metadata"""
    meta = [
        {
            "attributes": {"qualitePartie": "I", "typePersonne": "PP"},
            "identite": "Monsieur Amaury FOURET",
        },
        {
            "attributes": {"qualitePartie": "K", "typePersonne": "PP"},
            "identite": "Monsieur Romain GLE inconnu",
        },
        {
            "attributes": {"qualitePartie": "K", "typePersonne": "PM"},
            "identite": "S.A.R.L. LE BON BURGER",
        },
    ]
    metadata = process_metadata(meta, "jurica", model, tokenizer)

    assert isinstance(metadata, pd.DataFrame)

    csv_metadata = metadata.to_csv(index=False).replace("\r", "")

    assert csv_metadata == (
        "text,entity\n"
        "Amaury,personnePhysique\n"
        "FOURET,personnePhysique\n"
        "Romain,personnePhysique\n"
        "GLE,personnePhysique\n"
    ), csv_metadata


def test_process_metadata_none():
    """Testing preprocessing of empty metadata"""
    meta = None
    metadata = process_metadata(meta, "jurica", model, tokenizer)
    assert metadata is None
