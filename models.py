from pydantic import BaseModel, validator
from typing import Union, List, Dict
from juritools.type import NamedEntity


class Decision(BaseModel):
    idDocument: int
    text: str
    source: Union[str, None] = None
    # TODO: Spécifier les valeurs que doivent prendre ces champs
    meta: Union[Union[List[List], List[Dict]], None] = None
    categories: Union[List[str], None] = None

    @validator("text")
    @classmethod
    def text_should_not_be_empty(cls, v):
        if v == "":
            raise ValueError("text field is empty")
        return v


class NERResponse(BaseModel):
    tagger: List[NamedEntity] = []
    postProcess: List[NamedEntity] = []
    output: List[NamedEntity] = []
    checklist = []
