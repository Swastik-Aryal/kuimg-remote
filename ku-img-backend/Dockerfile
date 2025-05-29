# MULTILAYER BUILD (TO MINIMIZE DOCKER IMAGE SIZE) ----------------------------

# BUILDER IMAGE TO INSTALL BASIC DEPENDENCIES
FROM python:3.9.7-slim-buster as builder
RUN apt-get update && \
    apt-get -y install net-tools libffi-dev python3-dev gcc g++

# FASTAPI BASE IMAGE
FROM python:3.9.7-slim-buster
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages/

# Install Chrome dependencies and Chrome itself
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    gnupg \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libxss1 \
    libappindicator3-1 \
    libasound2 \
    fonts-liberation \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages/

# INSTALL REQUIREMENTS
COPY ./app/requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt

# INSTALL punkt and stopwords for nltk
RUN python -m nltk.downloader punkt punkt_tab stopwords

# COPY APP
WORKDIR /app
COPY ./app /app

# ENVIRONMENT VARIABLES
ENV APP_PATH '/app'
ENV APP_NAME 'kuimg'
ENV REDIS_FS "redis-fs:6379"
ENV DISPLAY=:99  
# Needed for Chrome headless

# EXPOSE PORT 80
EXPOSE 80

# RUN
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80","--reload", "--log-level", "debug"]
