#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import textwrap
import os
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast




def setup_BERT():
    bert_model_choice = os.environ.get('BERT_MODEL', 'distilbert-base-uncased')
    bert_model = DistilBertForQuestionAnswering.from_pretrained(bert_model_choice)
    bert_tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_choice)

    def generate_answer_BERT(question, answer_text, bert_model, bert_tokenizer):
        '''
        Takes a `question` string and an `answer_text` string (which contains the
        answer), and identifies the words within the `answer_text` that are the
        answer. Prints them out.
        '''
        # ======== Tokenize ========
        # Apply the tokenizer to the input text, treating them as a text-pair.
        if len(question) > 512:
            question = question[0:512]
        if len(answer_text) > 512:
            answer_text = answer_text[0:512]

        inputs = bert_tokenizer(question, answer_text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        # ======== Evaluate ========
        # Run our example question through the model.
        outputs = bert_model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(
            answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        answer = bert_tokenizer.convert_tokens_to_string(
            bert_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return answer

    return bert_model,bert_tokenizer,generate_answer_BERT





