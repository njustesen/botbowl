FROM python:3.9

RUN pip install --upgrade pip

WORKDIR /home/botbowl
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
RUN python setup.py install

WORKDIR /home/

CMD ["python botbowl/examples/server_example.py"]
