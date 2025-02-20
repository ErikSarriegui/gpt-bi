"""
==============================
UTILS
==============================

Erik Sarriegui Perez, AuriLab, Feb 2025
"""
def split_tokenize_count(batch, tokenizer, block_size):
    """
    Divide y tokeniza un lote de textos, contando el número de tokens en cada chunk.

    Esta función toma un lote de textos, los tokeniza y los divide en chunks de tamaño `block_size`.  Además, calcula la máscara de atención y la cantidad de tokens para cada chunk.  Es útil para preprocesar datos para modelos de lenguaje como GPT-2, que tienen una longitud de contexto limitada.

    Args:
        batch (dict): Un diccionario que contiene el lote de textos. Se espera que tenga una clave "text" con una lista de strings.
        tokenizer: El tokenizer a utilizar para la tokenización.
        block_size (int): El tamaño máximo de cada chunk de tokens.

    Returns:
        dict: Un diccionario con las siguientes claves:
            - "input_ids": Una lista de listas, donde cada sublista contiene los IDs de los tokens de un chunk.
            - "attention_mask": Una lista de listas, donde cada sublista contiene la máscara de atención para un chunk.
            - "n_tokens": Una lista con el número de tokens en cada chunk.
    """
    tokenized = tokenizer(batch["text"], truncation = False, add_special_tokens = False)
    
    nuevos_input_ids = []
    attention_masks = []
    n_tokens = []
    
    for input_ids in tokenized["input_ids"]:
        chunks = [input_ids[i:i + block_size] for i in range(0, len(input_ids), block_size)]
        
        for chunk in chunks:
            nuevos_input_ids.append(chunk)
            attention_masks.append([1] * len(chunk))
            n_tokens.append(len(chunk))
    
    return {
        "input_ids": nuevos_input_ids,
        "attention_mask": attention_masks,
        "n_tokens": n_tokens
    }