#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import textwrap
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


def setup_T5():
    slow_t5_model_choice = os.environ.get('T5_MODEL', 'mrm8488/t5-base-finetuned-summarize-news')
    # choice = 't5-large'
    model_T5_slow = T5ForConditionalGeneration.from_pretrained(slow_t5_model_choice)
    tokenizer_T5_slow = T5Tokenizer.from_pretrained(slow_t5_model_choice)
    device = torch.device('cpu')

    def generate_T5_answer(question, context, model_T5_slow, tokenizer_T5_slow):
        source_encoding = tokenizer_T5_slow(
            question,
            context,
            max_length=396,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        generated_ids = model_T5_slow.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            num_beams=1,  # greedy search
            # max_length=80,
            repetition_penalty=2.5,
            # early_stopping=True,
            use_cache=True)
        preds = [
            tokenizer_T5_slow.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]
        return "".join(preds)

    return model_T5_slow,tokenizer_T5_slow,generate_T5_answer


