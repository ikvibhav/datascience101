import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
DB_LOCATION = "./database"
COLLECTION_NAME = "reviews"
COLLECTION_FILENAME = "reviews.csv"
MODEL_NAME = "mxbai-embed-large"


def create_vector_store(
    embedding_model_name: str = MODEL_NAME,
    collection_name: str = COLLECTION_NAME,
    db_location: str = DB_LOCATION,
) -> Chroma:
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings,
    )
    return vector_store


def add_to_vector_store_from_csv(
    vector_store: Chroma, csv_file: str = COLLECTION_FILENAME
):
    """
    Add documents to the vector store.
    """
    df = pd.read_csv(csv_file)

    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i),
        )

        ids.append(str(i))
        documents.append(document)

    vector_store.add_documents(documents, ids=ids)


def get_vector_store_retriever():
    """
    Get a retriever for the vector store.
    """
    # Initialize a Chroma vector store with Ollama embeddings
    vector_store = create_vector_store()

    # Check if the database already exists, if not, add documents from the CSV file
    if not os.path.exists(DB_LOCATION):
        add_to_vector_store_from_csv(vector_store, COLLECTION_FILENAME)
    return vector_store.as_retriever(search_kwargs={"k": 5})
