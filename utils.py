def calculate_steps(batch_size, gradient_accumulation_steps, total_tokens, block_size):
    total_batch_size = batch_size * gradient_accumulation_steps
    tokens_per_batch = total_batch_size * block_size
    return (20 * total_tokens) // tokens_per_batch