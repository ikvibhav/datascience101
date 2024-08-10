# Binary Classification

## Description
An application to compare binary classification algorithms on popular datasets

## Setup Instructions

### 1. Create a Virtual Environment
```sh
python3 -m venv --prompt binaryclassification venv
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
docker build -t binaryclassification .
```

#### Run the Docker container
```sh
docker run -p 8501:8501 binaryclassification
```
