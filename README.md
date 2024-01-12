# API permettant de requêter le modèle de Machine Learning

## Installation

Nous détaillons ici l'installation complète de l'environnement permettant de faire fonctionner l'API.

Il faut au préalable installer le tokenizer spécifique aux décisions de jusitice, [jurispacy-tokenizer](https://github.com/Cour-de-cassation/jurispacy-tokenizer) et [juritools](https://github.com/Cour-de-cassation/juritools), le moteur de pseudonymisation de la Cour de cassation, ainsi que ses dépendances.

Finalement, on installe les dépendances liées à l'API :

```sh
cd nlp-pseudonymisation-api
pip install -r requirements.txt
conda deactivate
```

## Lancer l'API

Pour lancer l'API, on peut utiliser le fichier `server.py`:

```sh
python server.py
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
    categories: Optional[list[str]] = None
```

Il renvoie un JSON contenant une liste d'entités ainsi que des mises en doute.

L'autre endpoint permet de calculer la loss d'un document après sa vérification par un agent.

Les exemples de requêtes ci-dessous sont effectués en Python 3.7

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
        "L'entité 'Dupont' appartenant à la classe 'personnePhysique' se retrouve également dans une autre classe, une vérification manuelle est nécessaire.",
    ]
}
```

## Tests

### Structure des tests

Les tests de l'API sont contenus dans deux fichiers du dossier `tests`:

- `test_app.py`: permet de tester les différents points de terminaison de l'API.

### Prérequis

Pour lancer les tests, il faut installer les librairies de développement:

```sh
cd nlp-pseudonymisation-api
pip install -r requirements-dev.txt
```

De plus, pour tester le modèle, il faut préciser le chemin vers le modèle utilisé dans la variable d'environnement `MODEL_JURICA`.

```sh
export MODEL_JURICA='models/new_categories_model.pt'
```

### Tester l'API en lançant automatiquement une instance

Si on souhaite lancer une instance de l'API et la tester directement, on peut simplement utiliser:

```sh
cd nlp-pseudonymisation-api
pytest tests
```

### Tester une instance de l'API en fonctionnement

Si on souhaite lancer une instance de l'API pré-existante (python ou container docker), on doit spécifier l'URL de l'API dans une variable d'environnement `API_URL`:

```sh
export API_URL='http://localhost:8081'
```

Puis on peut lancer les tests avec:

```sh
cd nlp-pseudonymisation-api
python tests/test_app.py
```

Pour supprimer la variable d'environnement `API_URL`, on fait simplement `unset API_URL`.
