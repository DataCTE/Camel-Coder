import os
import openai

os.environ["OPENAI_API_KEY"] =  'your-api-key'  # replace 'your-api-key' with your actual API key
openai.api_key = "your-api-key" # replace 'your-api-key' with your actual API key
serpapi_api_key="your-api-key"

conversation_directory = "./workspace/" # Change to disired Path

assistant_role_name = "Ai Expert"
user_role_name = "Project Lead"
task = "Create Automata. Automata's objective is to evolve into a fully autonomous, self-programming Artificial Intelligence system that uses LLMs such as openai and huggingface!"

TOKEN_LIMIT = 14000

word_limit = 50 # word limit for task brainstorming
