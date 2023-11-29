FROM python:3.8.10

RUN mkdir NIA
WORKDIR /NIA
COPY . /NIA
VOLUME /NIA
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

CMD python test.py