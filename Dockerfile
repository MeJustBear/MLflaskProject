FROM python:3.8-slim
COPY . /MLflaskProject
WORKDIR /MLflaskProject
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean
#RUN apt-get add --no-cache --update \
#    python3 python3-dev gcc \
#    gfortran musl-dev g++ \
#    libffi-dev openssl-dev \
#    libxml2 libxml2-dev \
#    libxslt libxslt-dev \
#    libjpeg-turbo-dev zlib-dev
RUN pip install cython
RUN pip install --upgrade cython
RUN pip install --upgrade pip
RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python","app/app.py"]





