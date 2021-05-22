from google.cloud import storage
from tensorflow.keras.models import load_model
import numpy as np
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


encoder_model = None
decoder_model = None
word2idx = None
idx2word = None

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "UNK"
MAXLEN_INPUT = 439

maxlen_title = 48


def prepare():
    global encoder_model, decoder_model
    if encoder_model == None and decoder_model == None:

        encoder_model = load_model('encoder')
        decoder_model = load_model('decoder')


def load_dict():
    global word2idx, idx2word

    if word2idx == None:
        word2idx = {}
        with open("word.txt", "r") as f:
            for line in f.readlines():
                word2idx[line.replace("\n", "")] = len(word2idx) + 1

        idx2word = {v: k for k, v in word2idx.items()}


def decode_sequence(input_seq):
    global encoder_model, decoder_model

    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = word2idx[START_TOKEN]

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = idx2word[sampled_token_index]

        if(sampled_token != END_TOKEN):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == END_TOKEN or len(decoded_sentence.split()) >= (maxlen_title - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def format_input(sentence):
    from unicodedata import normalize
    from pythainlp.tokenize import word_tokenize
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    UNK_TOKEN = "UNK"
    MAXLEN_INPUT = 439

    def clean_input(sen):
        sen = normalize("NFKD", sen.strip().lower())
        sen = " ".join(sen.split())
        return sen

    # split number
    def is_num(word):
        return word.replace(",", "").replace(".", "").isnumeric()

    def clear_after_token(sen):
        newSentence = []
        for word in sen:
            word = word.strip()
            if is_num(word):
                word = "~".join(list(word))

            word = word.replace("(", "~(~").replace(")", "~)~").replace(
                "–", "-"). replace("-", "~-~").replace("?", "~?~")
            word = word.replace("“", '~"~').replace("”", '~"~')
            word = word.replace("‘", "~'~").replace("’", "~'~")
            word = word.strip().split('~')
            newSentence += word
        return newSentence

    def preprocess_for_keras(sen):
        sen = [START_TOKEN] + sen + [END_TOKEN]
        load_dict()
        sen = [word2idx.get(word, 1) for word in sen]
        sen = pad_sequences([sen], maxlen=MAXLEN_INPUT, dtype='int32',
                            padding='post', truncating='post', value=0)
        sen = sen.reshape(1, MAXLEN_INPUT)
        return sen

    return preprocess_for_keras(clear_after_token(word_tokenize(clean_input(sentence), engine="newmm")))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Body(BaseModel):
    text: str


@app.post('/predict')
def predict(body: Body):
    prepare()
    text = format_input(body.text)
    return decode_sequence(text)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=os.getenv('PORT', 8000))
