import pkg_resources
from pkg_resources import Requirement
import os
from datetime import datetime
from juritools.main import ner
from juritools.type import Decision
from flair.models import SequenceTagger
from jurispacy_tokenizer import JuriSpacyTokenizer
import config


def get_juritools_info():
    "Get version and date of juritools package"

    juritools_distribution = pkg_resources.working_set.resolve([Requirement("juritools")])[0]

    location = os.path.join(juritools_distribution.location, juritools_distribution.key)

    date = str(datetime.fromtimestamp(os.stat(location).st_mtime))
    version = juritools_distribution.version

    return date, version


def process_ner(
    decision: Decision,
    tokenizer: JuriSpacyTokenizer,
    model: SequenceTagger,
):
    try:
        return ner(
            decision=decision,
            tokenizer=tokenizer,
            model=model,
        )
    except Exception as exc:
        config.processing_request = False
        raise exc
