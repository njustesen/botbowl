FROM python:3.9

COPY . /home/botbowl

RUN pip install --upgrade pip
RUN pip install -e /home/botbowl/
RUN pip install -r /home/botbowl/requirements.txt
