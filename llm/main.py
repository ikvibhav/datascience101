from transformers import pipeline
import torch

# Check if GPU is available and print the number of GPUs
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.device_count())  # Check the number of GPUs available
print(torch.cuda.get_device_name(0))  # Get the name of the first GPU


model = pipeline(task="summarization", model="facebook/bart-large-cnn", device=0)  # Use GPU if available
response = model("tech is the future of humanity. It is the key to solving many of the world's problems, from climate change to disease. By harnessing the power of technology, we can create a better world for ourselves and future generations.")
print(response)