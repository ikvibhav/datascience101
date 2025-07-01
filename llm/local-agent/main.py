from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("--" * 20)
    user_input = input("Enter reviews (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    reviews = retriever.invoke(user_input)
    result = chain.invoke({"reviews": reviews, "question": user_input})
    print(result)
