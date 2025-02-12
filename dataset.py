from torch.utils.data import IterableDataset
from datasets import load_dataset
import itertools

SUBSETS = [
    "euscrawl-v1.1",
    "egunkaria",
    "booktegi",
    "wikipedia",
    "culturax",
    "colossal-oscar",
    "hplt-v1"
]

class LatxaDataset(IterableDataset):
    def __init__(self, tokenizer, block_size, split, batch_size, subsets = SUBSETS):
        self.split = split
        self.subsets = subsets
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size

    def __iter__(self):
        # Abrimos en streaming cada subset y los concatenamos
        iterators = [
            load_dataset("HiTZ/latxa-corpus-v1.1", subset, split=self.split, streaming=True)
            for subset in self.subsets
        ]
        combined = itertools.chain(*iterators)

        buffer = []
        batch = []

        # Acumular ejemplos en batch
        for example in combined:
            batch.append(example["text"])
            if len(batch) >= self.batch_size:
                # Tokenizamos el batch de textos
                tokenized_batch = self.tokenizer(batch, truncation=False)["input_ids"]
                # Procesamos cada ejemplo tokenizado
                for tokens in tokenized_batch:
                    buffer.extend(tokens)
                    # Cuando el buffer tiene suficientes tokens, seccionamos en bloques
                    while len(buffer) >= self.block_size:
                        block = buffer[:self.block_size]
                        buffer = buffer[self.block_size:]
                        yield {"input_ids": block, "labels": block.copy()}
                batch = []  # Reiniciamos el batch

        # Procesamos los ejemplos restantes (si los hubiera)
        if batch:
            tokenized_batch = self.tokenizer(batch, truncation=False)["input_ids"]
            for tokens in tokenized_batch:
                buffer.extend(tokens)
                while len(buffer) >= self.block_size:
                    block = buffer[:self.block_size]
                    buffer = buffer[self.block_size:]
                    yield {"input_ids": block, "labels": block.copy()}