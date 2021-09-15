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

slow_t5_model_choice = os.environ.get('T5_MODEL', 'mrm8488/t5-base-finetuned-summarize-news')
# choice = 't5-large'
model_T5_slow = T5ForConditionalGeneration.from_pretrained(slow_t5_model_choice)
tokenizer_T5_slow = T5Tokenizer.from_pretrained(slow_t5_model_choice)
device = torch.device('cpu')


def answer_question_T5(question, context):
    # choice ='t5-base'

    #     t5_prepared_Text = "question: " + question + context
    t5_prepared_Text = "question: " + question + "context: " + context
    tokenized_text = tokenizer_T5_slow.encode(t5_prepared_Text, return_tensors="pt").to(device)

    qa_ids = model_T5_slow.generate(tokenized_text,
                                    num_beams=5,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

    output = tokenizer_T5_slow.decode(qa_ids[0], skip_special_tokens=True)

    # print("\nAnswer: \n", textwrap.fill(output, 72) + '\n')
    return output

def generate_T5_answer(question,context):
  source_encoding=tokenizer_T5_slow(
      question,
      context,
      max_length = 396,
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
      #max_length=80,
      repetition_penalty=2.5,
      #early_stopping=True,
      use_cache=True)
  preds = [
          tokenizer_T5_slow.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
          for generated_id in generated_ids
  ]
  return "".join(preds)

