FROM python:3.8-slim
COPY . /MLflaskProject
WORKDIR /MLflaskProject
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean
RUN pip install cython
RUN pip install --upgrade cython
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python","app/run.py"]





