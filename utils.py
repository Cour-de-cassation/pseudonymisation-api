import logging
import subprocess
from typing import List

import pandas as pd
from fastapi import HTTPException
from flair.data import Sentence
from flair.models import SequenceTagger
from jurispacy_tokenizer import JuriSpacyTokenizer
from juritools.postprocessing import PostProcessFromEntities
from juritools.postprocessing import PostProcessFromSents
from juritools.postprocessing import PostProcessFromText
from juritools.predict import JuriTagger

from models import Decision, NERResponse


def replace_specific_encoding(text):
    """Replace End of Line tokens"""
    return text.replace("\f", "\n").replace("\r", "\n")


def get_juritools_info():
    "Get version and date of juritools package"
    cp = subprocess.run(
        "/home/juritools/anaconda3/bin/pipbyday | grep ' juritools'",
        shell=True,
        capture_output=True,
        text=True,
    )
    info_list = cp.stdout.split()
    return info_list[0], info_list[4]


def ner(decision: Decision, tokenizer: JuriSpacyTokenizer, model: SequenceTagger):
    """Returns the predictions of the NER Model

    Args:
        decision (Decision): the decision to analyze
        tokenizer (JuriSpacyTokenizer): tokenizer to create tokens and sentences
        model (SequenceTagger): a trained NER model

    Raises:
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    try:
        response = NERResponse()
        # SequenceTagger predictions
        text_decision = replace_specific_encoding(decision.text)

        juritag = JuriTagger(tokenizer, model)
        juritag.predict(text_decision)
        prediction_jsonified = juritag.get_entity_json_from_flair_sentences()
        response.tagger = prediction_jsonified
        # Process metadata
        metadata = process_metadata(decision.meta, decision.source, model, tokenizer)

        # Postprocessing on court decision text
        postpro_text = PostProcessFromText(
            text_decision, prediction_jsonified, manual_checklist=[], metadata=metadata
        )
        # Postprocessing on text
        postpro_text.manage_quote()
        postpro_text.manage_le()
        # Postprocessing on entities
        postpro_entities = PostProcessFromEntities(
            postpro_text.entities,
            manual_checklist=[],
            metadata=metadata,
            tokenizer=tokenizer,
        )
        # postpro_entities.match_physicomorale()
        postpro_entities.match_address_in_moral()
        if decision.categories and "personneMorale" not in decision.categories:
            postpro_entities.match_natural_persons_in_moral(False)
        postpro_entities.change_pro_to_physique()
        # postpro_entities.manage_natural_persons()
        postpro_entities.manage_year_in_date()
        postpro_entities.check_len_entities()  # personnephysique by default
        postpro_entities.check_entities()  # personnephysique by default
        postpro_entities.match_localite_in_adress()

        # Go back on postprocessing on text
        postpro_text.entities = postpro_entities.entities
        postpro_text.manual_checklist = postpro_entities.manual_checklist
        postpro_text.match_from_category(
            [
                "personnephysique",
                "professionnelavocat",
                "professionnelmagistratgreffier",
                "datedeces",
            ]
        )
        postpro_text.match_regex()
        postpro_text.juvenile_facility_entities()
        postpro_text.match_name_in_website()
        try:
            if metadata is not None:
                if decision.source == "jurinet":
                    postpro_text.match_metadata_jurinet()
                elif decision.source == "jurica":
                    postpro_text.match_metadata_jurica()
        except Exception:
            pass
        postpro_text.check_cadastre()

        # Postprocessing on flair sentences
        postpro_sents = PostProcessFromSents(
            juritag.flair_sentences,
            postpro_text.entities,
            manual_checklist=postpro_text.manual_checklist,
            metadata=metadata,
        )
        postpro_sents.apply_methods(change_pro_no_context=False)
        if decision.categories and "personneMorale" not in decision.categories:
            postpro_sents.match_cities_in_moral(False)

        entities = postpro_sents.ordered_entities()
        response.postProcess = entities
        # Handle categories parameter
        if isinstance(decision.categories, List):
            filter_entities = [
                entity
                for entity in postpro_sents.map_category_to_camelcase(entities)
                if entity.label in decision.categories
            ]
            response.output = filter_entities
        else:
            response.output = entities
        response.checklist = postpro_sents.manual_checklist

        return response

    except Exception as e:

        global processing_request
        processing_request = False

        logging.exception(f"{decision.idDocument} trace: {e}")

        raise HTTPException(
            status_code=400,
            detail=f"Failure of document id {decision.idDocument} "
            f"from {decision.source} with following trace: {e}",
        ) from e


def process_metadata(metadata, source, model, tokenizer):
    """
    This function processes the metadata coming from jurinet/jurica
    """

    if metadata is not None and len(metadata) > 1 and source == "jurinet":
        try:
            m = metadata
            col_list = [
                "ID_DOCUMENT",
                "TYPE_PERSONNE",
                "ID_PARTIE",
                "NATURE_PARTIE",
                "TYPE_PARTIE",
                "ID_TITRE",
                "NOM",
                "PRENOM",
                "NOM_MARITAL",
                "AUTRE_PRENOM",
                "ALIAS",
                "SIGLE",
                "DOMICILIATION",
                "LIG_ADR1",
                "LIG_ADR2",
                "LIG_ADR3",
                "CODE_POSTAL",
                "NOM_COMMUNE",
                "NUMERO",
            ]
            metadata = pd.DataFrame(m, columns=col_list)
        except Exception as exc:
            logging.exception("Exception occurred:")
            raise HTTPException(
                status_code=400,
                detail="Metadata are not in the good format.",
            ) from exc
    elif metadata is not None and len(metadata) > 1 and source == "jurica":
        try:
            m = metadata
            text = []
            ent = []
            flair_sentences = [
                Sentence(dico["identite"], use_tokenizer=tokenizer)
                for dico in m
                if dico["attributes"]["typePersonne"] == "PP"
            ]

            model.predict(flair_sentences)
            for sent in flair_sentences:
                for span in sent.get_spans(type="ner"):
                    if span.get_label("ner").value in ["personnePhysique"]:
                        text.append(span.text)
                        ent.append(span.get_label("ner").value)
            metadata = pd.DataFrame(data={"text": text, "entity": ent})
            metadata.drop_duplicates(inplace=True)
        except Exception as exc:
            logging.exception("Exception occurred:")
            raise HTTPException(
                status_code=400,
                detail="Metadata are not in the good format.",
            ) from exc
    else:
        metadata = None

    return metadata
