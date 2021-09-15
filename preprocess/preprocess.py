#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Preprocessing pipeline for Quasi QA."""

import argparse
from io import StringIO, BytesIO
import apache_beam as beam
import nltk
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.io.fileio import MatchAll, ReadMatches
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
import json
import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import logging
from nltk import word_tokenize

nltk.download('punkt')


def convert_pdf(readable_file):
    password = ''
    output_string = StringIO()
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
    fp = BytesIO(readable_file.read())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    maxpages = 0
    caching = True
    pagenos = set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = output_string.getvalue()
    text = text.replace(' \n', '\n')
    fp.close()
    device.close()
    output_string.close()
    return text


def create_paragraph_record(paragraph_text):
    tokens = nltk.word_tokenize(paragraph_text)
    sentence_tokens = nltk.sent_tokenize(paragraph_text)
    return {
        'paragraph': paragraph_text,
        'char': len(paragraph_text),
        'num_word_tokens': len(tokens),
        'sentences': sentence_tokens
    }


def run(argv=None):
    """Main entry point; defines and runs the preprocessing pipeline."""


    nltk.download('punkt')
    table_schema = bigquery.TableSchema()

    paragraph_schema = bigquery.TableFieldSchema()
    paragraph_schema.name = 'paragraph'
    paragraph_schema.type = 'string'
    paragraph_schema.mode = 'nullable'
    table_schema.fields.append(paragraph_schema)

    char_schema = bigquery.TableFieldSchema()
    char_schema.name = 'char'
    char_schema.type = 'integer'
    char_schema.mode = 'nullable'
    table_schema.fields.append(char_schema)

    num_token_schema = bigquery.TableFieldSchema()
    num_token_schema.name = 'num_word_tokens'
    num_token_schema.type = 'integer'
    num_token_schema.mode = 'nullable'
    table_schema.fields.append(num_token_schema)

    # kb_id_schema = bigquery.TableFieldSchema()
    # kb_id_schema.name = 'kb_id'
    # kb_id_schema.type = 'string'
    # kb_id_schema.mode = 'nullable'
    # table_schema.fields.append(kb_id_schema)

    num_token_schema = bigquery.TableFieldSchema()
    num_token_schema.name = 'sentences'
    num_token_schema.type = 'string'
    num_token_schema.mode = 'repeated'
    table_schema.fields.append(num_token_schema)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--output',
        dest='output',
        required=False,
        help='Output file to write results to.')
    parser.add_argument(
            '--pubsubtopic',
        dest='pubsubtopic',
        required=False,
        help='Output file to write results to.')
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(
        pipeline_args, streaming=True,save_main_session=True
    )

    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(options=pipeline_options) as p:
        # Read the text file[pattern] into a PCollection.
        # files = p | 'ListFiles' >> MatchFiles(known_args.input
        files = p | "ReadfromPubSub" >> beam.io.ReadFromPubSub(topic=known_args.pubsubtopic)

        paragraphs = (files | "GetFileNames" >> beam.Map(lambda x: json.loads(x)[
            'url']) | 'ListFiles' >> MatchAll() | "Read files" >> ReadMatches() | 'ReadPdf' >> beam.Map(
            convert_pdf) | 'ConvertToParagraphs' >> beam.FlatMap(
            lambda x: x.split('\n\n')) | 'FilterNulls' >> beam.Filter(
            lambda x: len(x) > 0) | 'CreateRecords' >> beam.Map(create_paragraph_record))
        _ = paragraphs | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(
            'qqatext_df', dataset='qqa_text',
            schema=table_schema,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()