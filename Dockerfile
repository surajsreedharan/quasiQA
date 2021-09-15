FROM python:3.6

COPY . .

RUN mkdir data

RUN pip install -r requirements.txt

RUN python -m spacy download fr_core_news_sm


ENTRYPOINT ["sh","/start.sh"]