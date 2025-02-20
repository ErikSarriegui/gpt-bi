from transformers import GPT2TokenizerFast
import huggingface_hub

from dataset import loadFullCorpus


def get_texts(dataset):
    for item in dataset:
        yield item["text"]

def train_tokenizer() -> None:
    """
    Entrena un tokenizer basado en el de GPT-2 utilizando un dataset combinado de
    Latxa y Wikipedia en español.

    Esta función configura y ejecuta el entrenamiento del tokenizador. Utiliza el
    dataset combinado Latxa y Wikipedia, y emplea la librería `transformers` de
    Hugging Face.

    ==========================
    SET UP ENVIROMENT
    ==========================
    """
    huggingface_hub.login(token = "<token>")

    """
    ==========================
    DEFINING THE TRAINING
    ==========================
    """
    BASE_TOKENIZER_ID = "gpt2"
    VOCAB_SIZE = 32000

    PUSH_REPO_ID = "gpt-bi"
    PUSH_ORGANIZATION = "AuriLab"

    SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    BOS_TOKEN = "<|im_start|>"
    EOS_TOKEN = "<|im_end|>"
    PAD_TOKEN = "<|im_end|>"
    UNK_TOKEN = "<|endoftext|>"

    CHAT_TEMPLATE = ("{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}" 
        "{{ '<|im_start|>system\nAIko laguntzaile lagungarria zara, GPT-Bi izenekoa, AuriLab-ek hezia.<|im_end|>\n' }}" 
        "{% endif %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
    
    """
    ==========================
    TRAIN THE TOKENIZER
    ==========================
    """
    dataset = loadFullCorpus()

    base_tokenizer = GPT2TokenizerFast.from_pretrained(BASE_TOKENIZER_ID)

    tokenizer = base_tokenizer.train_new_from_iterator(get_texts(dataset["train"]), vocab_size = VOCAB_SIZE)

    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.bos_token = BOS_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.unk_token = UNK_TOKEN

    tokenizer.chat_template = CHAT_TEMPLATE

    tokenizer.push_to_hub(repo_id = PUSH_REPO_ID, organization = PUSH_ORGANIZATION)


if __name__ == "__main__":
    """
    ==========================
    LAUNCH TRAINING
    ==========================
    """
    train_tokenizer()