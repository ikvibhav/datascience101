from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_db import get_vector_store_retriever


class PizzaQAAssistant:
    """Handles pizza restaurant Q&A using LLM and vector retriever."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model = OllamaLLM(model=model_name)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in answering questions about a pizza restaurant.

            Here are some relevant reviews: {reviews}

            Here is the question to answer: {question}
            """
        )
        self.chain = self.prompt | self.model
        self.retriever = get_vector_store_retriever()

    def answer(self, user_input: str) -> str:
        reviews = self.retriever.invoke(user_input)
        return self.chain.invoke({"reviews": reviews, "question": user_input})


def main():
    assistant = PizzaQAAssistant()
    print("Pizza Restaurant Q&A Assistant. Type 'exit' to quit.")
    while True:
        print("--" * 20)
        user_input = input("Enter your question: ")
        if user_input.lower() == "exit":
            break
        try:
            result = assistant.answer(user_input)
            print(result)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
