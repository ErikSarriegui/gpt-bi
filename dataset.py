"""
==============================
DATASET
==============================

Erik Sarriegui Perez, AuriLab, Feb 2025
"""
from datasets import load_dataset, DatasetDict, concatenate_datasets

LATXA_DATASET_ID = "HiTZ/latxa-corpus-v1.1"
LATXA_SUBSETS = ["euscrawl-v1.1", "egunkaria", "booktegi", "wikipedia", "culturax", "colossal-oscar", "hplt-v1"]

WIKIPEDIA_DATASET_ID = "wikimedia/wikipedia"
WIKIPEDIA_SUBSET = "20231101.es"

def loadLatxaDataset() -> DatasetDict:
    """
    Carga el dataset Latxa, combinando los subconjuntos especificados y separando los datos en conjuntos de entrenamiento y prueba.

    El dataset Latxa se compone de varios subconjuntos.  Esta función itera sobre ellos, cargando cada uno y combinando los conjuntos
    'test' y 'validation' en un único conjunto de prueba.  Finalmente, concatena todos los conjuntos de entrenamiento y todos los
    conjuntos de prueba para crear un único DatasetDict.

    Returns:
        - DatasetDict: Un diccionario que contiene los conjuntos de entrenamiento y prueba del dataset Latxa.
    """
    train_datasets, test_datasets = [], []

    for subset in LATXA_SUBSETS:
        dataset = load_dataset(LATXA_DATASET_ID, subset)
        
        processed_test = concatenate_datasets([dataset["test"], dataset["validation"]])
        
        train_datasets.append(dataset["train"])
        test_datasets.append(processed_test)

    return DatasetDict({
        "train": concatenate_datasets(train_datasets),
        "test": concatenate_datasets(test_datasets)
    })

def loadWikipediaDataset() -> DatasetDict:
    """
    Carga el dataset de Wikipedia en español, lo divide en conjuntos de entrenamiento y prueba, y elimina columnas innecesarias.

    Carga una porción del dataset de Wikipedia en español, específicamente la versión con fecha '20231101.es'.  Selecciona los
    primeros 97000 (Es lo que falta para llegar al óptimo de tokens según Chinchilla) ejemplos para el conjunto inicial. Luego,
    elimina las columnas 'id', 'url' y 'title', ya que no son necesarias. Finalmente, divide el dataset resultante en conjuntos
    de entrenamiento y prueba, asignando el 2% al conjunto de prueba (es el mismo porcentaje que Latxa).

    Returns:
        - DatasetDict: Un diccionario que contiene los conjuntos de entrenamiento y prueba del dataset de Wikipedia.
    """
    wikipedia_dataset = load_dataset(WIKIPEDIA_DATASET_ID, WIKIPEDIA_SUBSET, split = f"train[:97000]")
    wikipedia_dataset = wikipedia_dataset.remove_columns(["id", "url", "title"])
    return wikipedia_dataset.train_test_split(test_size = 0.02)

def loadFullCorpus() -> DatasetDict:
    """
    Carga los datasets Latxa y Wikipedia en Español, y los combina en un único DatasetDict.

    Esta función utiliza las funciones `loadLatxaDataset()` y `loadWikipediaDataset()` para cargar los datasets Latxa y Wikipedia
    respectivamente. Luego, concatena los conjuntos de entrenamiento de ambos datasets en un único conjunto de entrenamiento, y
    de manera similar, concatena los conjuntos de prueba en un único conjunto de prueba.

    Returns:
        DatasetDict: Un diccionario que contiene los conjuntos de entrenamiento y prueba combinados de Latxa y Wikipedia.
    """
    latxa_dataset = loadLatxaDataset()
    wikipedia_dataset = loadWikipediaDataset()

    return DatasetDict({
        "train": concatenate_datasets([latxa_dataset["train"], wikipedia_dataset["train"]]),
        "test": concatenate_datasets([latxa_dataset["test"], wikipedia_dataset["test"]])
    })