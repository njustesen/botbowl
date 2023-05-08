FROM python:3.9

WORKDIR /home/botbowl
COPY . .

RUN pip install --upgrade pip

RUN pip install -e .
RUN python setup.py install

WORKDIR /home/

ENTRYPOINT ["python", "botbowl/examples/server_example.py"]
