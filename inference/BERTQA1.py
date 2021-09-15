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

bert_model_choice =  os.environ.get('BERT_MODEL', 'distilbert-base-uncase')
bert_model = DistilBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
bert_tokenizer = DistilBertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
def answer_question_BERT(question, answer_text):
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
        
    input_ids = bert_tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    #print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(bert_tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    output_score = bert_model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                          token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(output_score.start_logits)
    answer_end = torch.argmax(output_score.end_logits)

    # Get the string versions of the input tokens.
    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

#     print('Answer: "' + answer + '"')
#     print('~~~')
    return answer


def generate_answer_BERT(question, answer_text):
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
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    answer = bert_tokenizer.convert_tokens_to_string(bert_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer