import joblib
import warnings
from typing import Union
import numpy as np
from fastapi import FastAPI, Query
import uvicorn
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


warnings.filterwarnings("ignore")

SEED = 34
tf.random.set_seed(SEED)
MAX_LEN = 70


tokenizer = GPT2Tokenizer.from_pretrained("Narsil/gpt3")
GPT2 = TFGPT2LMHeadModel.from_pretrained("Narsil/gpt3", pad_token_id=tokenizer.eos_token_id)

def gen_sent(example_text):
    example_text = str(example_text)
    input_ids = tokenizer.encode(example_text, return_tensors='tf')
    greedy_output = GPT2.generate(input_ids, max_length = MAX_LEN)
    return (tokenizer.decode(greedy_output[0], skip_special_tokens = True))


app = FastAPI()
@app.get("/make_predict")
def read_items(q: Union[str, None] = None):
    try:
        res = gen_sent(q)
    except Exception as E:
        res = f"there's an error: {E}"
    return {"res": res}

if __name__ == "__main__":
	uvicorn.run("test_wazuh_model:app", host="0.0.0.0")
