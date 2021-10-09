from flask import Flask
from logging.config import dictConfig
import os


dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


app = Flask(__name__)

env_bigquery_df = os.environ.get("BIGQUERY_DF")

model_type = os.environ.get("MODEL_TYPE")
if model_type.lower()=='fastt5':
    from inference.fastT5QA import setup_fastT5
    model,tokenizer,answerer = setup_fastT5()
elif model_type.lower()=='bert':
    from inference.BERTQA1 import setup_BERT
    model,tokenizer,answerer = setup_BERT()
else:
    from inference.T5QA import setup_T5
    model, tokenizer, answerer = setup_T5()

from inference.config import get_config

app.config.from_object(get_config())
from inference import api

