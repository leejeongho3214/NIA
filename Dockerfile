FROM ubuntu:20.04
FROM python:3.8.10

RUN mkdir NIA
WORKDIR /NIA
COPY . /NIA
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD python main.py