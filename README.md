# API permettant de requêter le modèle de Machine Learning

## Installation

Nous détaillons ici l'installation de l'API.


Il faut au préalable installer le tokenizer spécifique aux décisions de jusitice, [jurispacy-tokenizer](https://github.com/Cour-de-cassation/jurispacy-tokenizer) et [juritools](https://github.com/Cour-de-cassation/juritools), le moteur de pseudonymisation de la Cour de cassation, ainsi que ses dépendances.

Finalement, on installe les dépendances liées à l'API :
```console
$ cd nlp-pseudonymisation-api
$ pip install -r requirements.txt
```

## Lancer l'API

```console
$ cd nlp-pseudonymisation-api
$ python server.py
```

## Exemple de requêtes

L'API possède deux endpoints principaux. Le endpoint /docs permet de les retrouver et les tester.

Le endpoint effectuant la pseudonymisation est /ner. Celui-ci accepte en entrée un JSON respectant le schéma de données suivant :

```python
class Decision(BaseModel):
    idDocument: int
    text: str
    source: Optional[str] = None
    meta: Optional[str] = None
    categories: Optional[List[str]] = None
```

Il renvoie un JSON contenant une liste d'entités ainsi que des mises en doute.

L'autre endpoint permet de calculer la loss d'un document après sa vérification par un agent.

Les exemples de requêtes ci-dessous sont effectués en Python 3.9.

### **Exemple de retour au format json d'une requête sur le endpoint /ner**

```json
{
    "entities": [
        {
            "text": "DUPONT",
            "start": 844,
            "end": 850,
            "label": "personnePhysique",
            "source": "NER model"
        },
        {
            "text": "Dupont",
            "start": 1227,
            "end": 1243,
            "label": "professionnelMagistratGreffier",
            "source": "NER model"
        },
        {
            "text": "Paris",
            "start": 2111,
            "end": 2116,
            "label": "localite",
            "source": "postprocess"
        },
    ],
    "check_needed": true,
    "checklist": [
        "L'annotation 'Dupont' est de catégorie 'personnePhysique' mais on retrouve la même annotation dans une autre catégorie 'professionnelMagistratGreffier'. Les annotations sont-elles réellement de catégories différentes ?",
    ]
}
```

## Tests

Pour lancer les différents tests unitaires de l'API:

```console
$ pip install pytest
$ cd nlp-pseudonymisation-api
$ pytest
```