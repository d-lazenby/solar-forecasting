FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

RUN mkdir src

COPY ["src/inference.py", "src/preprocessing.py", "src/"]

RUN mkdir models

COPY ["models/final_model.bin", "models/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "src.inference:app"]