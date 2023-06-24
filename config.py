import os
import openai

os.environ["OPENAI_API_KEY"] =  'your-api-key'  # replace 'your-api-key' with your actual API key
openai.api_key = "your-api-key" # replace 'your-api-key' with your actual API key
serpapi_api_key="your-api-key"

folder_path = "path/to/workspace/" # Change to disired Path


assistant_role_name = "Ai Expert"
user_role_name = "Project Lead"
task = "create a system that automatically produces a requested openai agent system for a user(not any spesfic one it must be felxiable)"

TOKEN_LIMIT = 14000

word_limit = 50 # word limit for task brainstorming
