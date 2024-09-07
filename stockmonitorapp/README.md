# Stock Monitor

## Description
An application to monitor and understand stocks

## Current Status
1. Plotting SnP500 company stocks
2. Plotting moving averages

## Setup Instructions

### 1. Create a Virtual Environment
```sh
python3 -m venv --prompt stocktracker venv
```

### 2. Activate the Virtual Environment
```sh
source venv/bin/activate
```

### 3. Install the packages from requirements.txt
```sh
pip install -r requirements.txt
```

## Running Instructions

### 1. Using Bash Terminal
```sh
streamlit run app.py
```

### 2. Using Docker

#### Build the Docker image
```sh
docker build -t stockmonitor .
```

#### Run the Docker container
```sh
docker run -p 8501:8501 stockmonitor
```
