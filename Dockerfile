FROM python:3.5

COPY requirements.txt /tmp
COPY data/ /tmp/
COPY trainer/ /tmp/

WORKDIR /tmp
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y vim