from inference import app,env_bigquery_df
import logging

from google.cloud import bigquery
import os
import collections
from google.cloud import storage
import re
import spacy
from pathlib import Path
import threading
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from flask import request, jsonify
import numpy as np
import pandas as pd
from inference import model,tokenizer,answerer
from google.cloud import pubsub_v1
import json
import uuid

global corpus_df
global para_df
global tfidf_transformer_sentences
global sentence_tfidf_embed

corpus_df = None
para_df = None
tfidf_transformer_sentences = None
sentence_tfidf_embed = None
stop_words = text.ENGLISH_STOP_WORDS
tfidf_transformer = TfidfVectorizer(stop_words=stop_words)


def parse_blob_path(path):
    """Parse a gcs path into bucket name and blob name
    Args:
        path (str): the path to parse.
    Returns:
        (bucket name in the path, blob name in the path)
    Raises:
        ValueError if the path is not a valid gcs blob path.
    Example:
        `bucket_name, blob_name = parse_blob_path('gs://foo/bar')`
        `bucket_name` is `foo` and `blob_name` is `bar`
    """
    match = re.match('gs://([^/]+)/(.+)$', path)
    if match:
        return match.group(1), match.group(2)
    raise ValueError('Path {} is invalid blob path.'.format(
        path))


def download_blob(source_blob_path, destination_file_path):
    """Downloads a blob from the bucket.

    Args:
        source_blob_path (str): the source blob path to download from.
        destination_file_path (str): the local file path to download to.
    """
    bucket_name, blob_name = parse_blob_path(source_blob_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=blob_name)  # Get list of files
    print("checking for folder >>", destination_file_path)
    if not os.path.exists(destination_file_path):
        os.makedirs(destination_file_path)
        print('created folder >>>', destination_file_path)
    for blob in blobs:
        file_name = destination_file_path + blob.name.replace(blob_name, '')
        filepaths = blob.name.replace(blob_name, '').split("/")
        directory = destination_file_path + '/' + "/".join(filepaths[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        print('copying file >>>>', file_name)
        blob.download_to_filename(file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_path,
        destination_file_path))

def CosineSimilarity( u, v):
    num = np.dot(u, v)
    den = (np.linalg.norm(u) * np.linalg.norm(v))
    if den > 0:
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    else:
        return 0




def error_template(error, code, type, message):
    """
    Generic error template for the api
    :param error: Error object
    :param code: HTTP Code
    :param type: Error Type
    :param message: Error message
    :return:
    """
    logging.error(error)
    success = False
    response = {
        'success': success,
        'error': {
            'type': type,
            'message': message
        }
    }

    return jsonify(response), code


def setup_model():
    global corpus_df
    global para_df
    global tfidf_transformer_sentences
    global sentence_tfidf_embed

    bqclient = bigquery.Client()
    query_sent = 'with pr as (select row_number() over() as par_num,sentences  from ' \
                 '`'+env_bigquery_df+'` where num_word_tokens>10) select pr.par_num, ' \
                 'sent  from pr,unnest(pr.sentences) as sent '
    query_para = 'select row_number() over() as par_num, paragraph  from `'+env_bigquery_df+'` ' \
                 'where num_word_tokens>10 '

    corpus_df = (
        bqclient.query(query_sent)
            .result()
            .to_dataframe()
    )

    para_df = (bqclient.query(query_para)
               .result()
               .to_dataframe())
    sentence_list = corpus_df['sent']
    # sentence_list = self.lemmatize()

    tfidf_transformer.fit(sentence_list)
    X_train_tfidf = tfidf_transformer.transform(sentence_list)
    tfidf_transformer_sentences = tfidf_transformer
    sentence_tfidf_embed = X_train_tfidf.toarray()


@app.route("/qqa/setup", methods=["GET"])
def load_model_enpoint():
    # th = threading.Thread(target=setup_model)
    # th.start()
    setup_model()
    response = {'load_status': 'MODEL_LOADED'}
    return jsonify(response), 200

@app.route("/qqa/reset", methods=["GET"])
def unload_model_enpoint():
    bqclient = bigquery.Client()
    query = 'DELETE FROM `'+env_bigquery_df+'` WHERE true'
    bqclient.query(query).result()
    response = {'load_status': 'MODEL_REMOVED'}
    return jsonify(response), 200

@app.route("/qqa/load_kb", methods=["POST"])
def trigger_preprocess():
    kb_id=str(uuid.uuid4())
    topic_id = os.environ.get('PUBSUB_TOPIC','pubsub-test')
    project_id = os.environ.get("GCLOUD_PROJECT",'gcloudproj-name')
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    data = request.get_json(force=True)
    # kb_path = data.get('kb_path').strip()
    # data = json.dumps({"url": kb_path,"kb_id":kb_id})
    kb_path = data.get('kb_path').strip()
    data = json.dumps({"url": kb_path})
    # Data must be a bytestring
    data = data.encode("utf-8")
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data)
    print('Published>>>>')
    print(future.result())
    response = {'load_status': 'KNOWLEDGE_LOAD_INITIATED','kb_id':kb_id}
    return jsonify(response), 200

def shortlistParagraph(question):
    query_embed = tfidf_transformer_sentences.transform([question]).toarray().T.flatten()
    sentence_similarity = [CosineSimilarity(x, query_embed) for x in sentence_tfidf_embed]
    corpus_df['Similarity'] = sentence_similarity
    similarity_max_df = pd.pivot_table(corpus_df, values='Similarity', index='par_num', aggfunc=max)
    top_n_paragraphs = similarity_max_df.nlargest(5, 'Similarity')
    return top_n_paragraphs


@app.route("/qqa/answer", methods=["POST"])
def pred_bert():
    data = request.get_json(force=True)
    question = data.get('input').strip()
    top_paragraphs = shortlistParagraph(question)
    json_reply = []
    num_para = data.get('number_of_paragraphs')
    for i in range(num_para):
        top_para = top_paragraphs.iloc[i]
        if top_para['Similarity'] >= 0.1:
            para = para_df.loc[para_df['par_num'] == top_para.name]['paragraph'].values[0]
            abstractive_answer = answerer(question, para,model,tokenizer)
            json_reply.append({'rank':i,'answer':abstractive_answer})
            print("\nAnswer:\n", abstractive_answer)
    res = {'answers':json_reply}
    return jsonify(res), 200


