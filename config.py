import os
import openai

os.environ["OPENAI_API_KEY"] =  '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'  # replace 'your-api-key' with your actual API key
openai.api_key = "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" # replace 'your-api-key' with your actual API key
serpapi_api_key="665359845488368a0aaff3cf2617ee20a052ac05a44824f2e65ada252beb9eb6"

conversation_directory = "/home/Xander/Documents/Work-documents/camel_coder/workspace/" # Change to disired Path

assistant_role_name = "Ai Expert"
user_role_name = "Project Lead"
task = "Create Automata. Automata's objective is to evolve into a fully autonomous, self-programming Artificial Intelligence system that uses LLMs such as openai and huggingface!"

TOKEN_LIMIT = 14000

word_limit = 50 # word limit for task brainstorming
