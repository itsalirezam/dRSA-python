# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2TokenizerFast, GPT2Model
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import sys

# for chunk_size: Llama set to 7000 and GPT set to 800
# seq_len max: Llama 8000? and GPT 1024

def get_embeddings_efficient_average(words, tokenizer, model, layer_to_take, chunk_size=800):
    """
    Returns one embedding per word.
    layers selected to average in the selected_layers variable
    """
    all_embeddings = []

    for i in tqdm(range(0, len(words), chunk_size), desc="Processing chunks"):

        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)

        enc = tokenizer(
            chunk_text,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        seq_len = enc['input_ids'].shape[1]
        if seq_len > 1024:
            raise Exception('batch too big, sticks out of model context window')

        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            selected_layer = hidden_states[layer_to_take]  # (batch_size, sequence_len(n_tokens), hidden_size)
            # selected_layers = torch.stack(hidden_states[-4:]).mean(0)

        offsets = enc.offset_mapping[0].tolist()
        emb_tokens = selected_layer[0]  # (n_tokens, d)

        word_embeddings = []
        temp = []

        # Iterate through token embeddings with offsets
        for (start, end), emb in zip(offsets, emb_tokens):
            temp.append(emb)

            # word boundary: offset ends exactly where a space occurs (your rule)
            if end < len(chunk_text) and chunk_text[end] == " ":
                word_embeddings.append(torch.stack(temp).mean(0))
                temp = []

        # Handle last word (no trailing space)
        if temp:
            word_embeddings.append(torch.stack(temp).mean(0))

        all_embeddings.extend(word_embeddings)

        # Optional cleanup
        if torch.cuda.is_available() and i % (chunk_size * 10) == 0:
            torch.cuda.empty_cache()

    return torch.stack(all_embeddings, dim=0)


# ========================== main

## load text

# s = "this is a sentence"
# word_list = s.split(" ")

df = pd.read_csv('podcast_transcript.csv')
word_list = df['word'].tolist()

## run

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=True)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
model = GPT2Model.from_pretrained("gpt2-xl", output_hidden_states=True)
model.eval()

selected_text = word_list # a list of words

layer_number = int(sys.argv[1])
print(f'layer number: {layer_number}')

word_embeddings = get_embeddings_efficient_average(selected_text, tokenizer, model, layer_to_take=layer_number-1)

print('word_embeddings shape: ', word_embeddings.shape)

word_embeddings_npa = np.array(word_embeddings)
# np.save(f'Llama_embedding_Layer{layer_number}', word_embeddings_npa)
np.save(f'GPT2XL_embedding_Layer{layer_number}', word_embeddings_npa)

# Llama: [3, 10, 16, 22, 29]
# GPT: 4 15 24 34 44

# run: python get_embeddings.py 15