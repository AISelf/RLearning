FROM python:3.7.6-stretch
# 3.7.6-alpine

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils default-libmysqlclient-dev gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN pip install zipline
RUN pip install numpy
RUN pip install gym
RUN pip install fastparquet snappy stable-baselines
RUN pip install tensorflow==1.15.0
RUN pip install sklearn
WORKDIR /ws
EXPOSE 8080
CMD python manage.py runserver 0.0.0.0:8080