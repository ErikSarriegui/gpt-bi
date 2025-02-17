from datasets import load_dataset, concatenate_datasets

"""
SUBSETS = [
    "euscrawl-v1.1",
    "egunkaria",
    "booktegi",
    "wikipedia",
    "culturax",
    "colossal-oscar",
    "hplt-v1"
]
"""

SUBSETS = [
    "booktegi"
]

def loadLatxa(split : str):
    train_dataset_full = [
        load_dataset("HiTZ/latxa-corpus-v1.1", subset, split = split)
        for subset in SUBSETS
    ]

    return concatenate_datasets(train_dataset_full)