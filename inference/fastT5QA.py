#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import print_function

import os

import numpy as np
import pandas as pd
import torch
import textwrap
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from fastT5 import (OnnxT5, get_onnx_runtime_sessions,
                    generate_onnx_representation, quantize)



def setup_fastT5():
    fast_t5_model = os.environ.get('FAST_T5_MODEL', 't5-small')
    onnx_model_paths = generate_onnx_representation(fast_t5_model)
    quant_model_paths = quantize(onnx_model_paths)

    model_sessions = get_onnx_runtime_sessions(quant_model_paths)

    model_T5 = OnnxT5(fast_t5_model, model_sessions)

    tokenizer_T5 = T5Tokenizer.from_pretrained(fast_t5_model)
    device = torch.device('cpu')

    def fast_generate_T5_answer(question, context, model_T5, tokenizer_T5):
        source_encoding = tokenizer_T5(
            question,
            context,
            max_length=396,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        generated_ids = model_T5.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            num_beams=1,  # greedy search
            # max_length=80,
            repetition_penalty=2.5,
            # early_stopping=True,
            use_cache=True)
        preds = [
            tokenizer_T5.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]
        return "".join(preds)
    
    return model_T5,tokenizer_T5,fast_generate_T5_answer


