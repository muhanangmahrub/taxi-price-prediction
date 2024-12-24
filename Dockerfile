FROM python:3.12
WORKDIR /

COPY taxi_price_training.py taxi_price_training.py
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
ENTRYPOINT [ "python3", "taxi_price_training.py" ]