FROM  python:3.11-slim

RUN pip install pandas numpy scikit-learn joblib

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME

ENTRYPOINT ["python", "inference.py"]
CMD ["file1", "file2"]
