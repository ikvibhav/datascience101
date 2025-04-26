from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

# Ignore warnings from the transformers library
set_verbosity_error()


def question_age_explanation():
    """
    This function uses a pre-trained model to generate an explanation for a given question.
    It takes in a topic and an age as input and generates a detailed explanation suitable for that age.
    """
    # Define a summarization pipeline having a facebook model
    model = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

    # Wrap it inside LangChain
    llm = HuggingFacePipeline(pipeline=model)

    # Define a prompt template for summarization
    template = PromptTemplate.from_template(
        "Summarize the following topic for a {age} year old: {topic}. "
    )

    # Create a langchain chain
    chain = template | llm

    # Get user input for topic and age
    topic = input("Enter the topic: ")
    age = input("Enter the age: ")

    # Execute the chain with the provided topic and age
    response = chain.invoke({"topic": topic, "age": age})

    # Print the generated explanation
    print(f"Explanation for a {age} year old about {topic}:")
    print(response)


if __name__ == "__main__":
    question_age_explanation()