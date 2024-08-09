# DataScience 101

## Description
A repository to document my learnings and code related to data science

## Setup Instructions

### 1. Create a Virtual Environment
```sh
python3 -m venv --prompt datascience venv
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
docker build -t datascience101 .
```

#### Run the Docker container
```sh
docker run -p 8501:8501 datascience101
```