FROM python:3.5-slim
MAINTAINER nehasinghsikerwar@gmail.com
USER root
WORKDIR /app
ADD . /app
RUN apt update && apt install --no-install-recommends -y python3-dev  gcc build-essential
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]