import os
import openai
import requests
import time
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from transformers import GPT2Tokenizer
from serpapi import GoogleSearch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

class ProblemSolver:
    def __init__(self, openai_key, serpapi_api_key):
        self.goal = None
        self.conversation = []
        self.vector_memory = []
        self.openai_key = openai_key
        self.serpapi_api_key = serpapi_api_key

    def truncate_tokens(self, text, max_tokens):
        tokens = tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]
        return tokenizer.decode(tokens)

    def set_goal(self, task, conversation):
        self.goal = task
        self.conversation = conversation  # store the entire conversation

    def brainstorm(self):
        # Use conversation data in the brainstorming method
        conversation_str = "\n".join([f"{role_name}: {msg}" for role_name, msg in self.conversation])
        conversation_str = self.truncate_tokens(conversation_str, 14000)
        # Update vector_memory with the conversation string
        self.vector_memory.append(conversation_str)
        # Now conversation_str can be used in brainstorming
        question = "Phase 1: Brainstorming\n\nPlease generate three or more potential solutions for the given problem."
        response = self.generate_response(question)
        self.update_memory(response)
        return response

    # Set the SerpApi API key as an environment variable
    def set_serpapi_api_key(self):
        os.environ["SERPAPI_API_KEY"] = self.serpapi_api_key

    def thought(self):
        conversation_str = "\n".join([f"{role_name}: {msg}" for role_name, msg in self.conversation])
        conversation_str = self.truncate_tokens(conversation_str, 14000)
        self.vector_memory.append(conversation_str)
        question = "Phase 2: Thought\n\nPlease think deeply about the potential solutions and generate a specific topic or question for a Google search query."
        response = self.generate_response(question)
        self.update_memory(response)

        # Perform Google search with the generated query
        google_results = self.google_search(response)

        # Perform the secondary step (Thought Second Step)
        thought_second_step = self.thought_second_step(google_results, response)
        self.update_memory(thought_second_step)
        return thought_second_step

    def evaluate(self):
        conversation_str = "\n".join([f"{role_name}: {msg}" for role_name, msg in self.conversation])
        conversation_str = self.truncate_tokens(conversation_str, 14000)
        self.vector_memory.append(conversation_str)
        question = "Phase 3: Evaluation\n\nPlease objectively assess each potential solution by considering their pros and cons, initial effort, implementation difficulty, potential challenges, and expected outcomes. Assign a probability of success and a confidence level for each option."
        response = self.generate_response(question)
        self.update_memory(response)
        return response

    def thought_expand(self):
        conversation_str = "\n".join([f"{role_name}: {msg}" for role_name, msg in self.conversation])
        conversation_str = self.truncate_tokens(conversation_str, 14000)
        self.vector_memory.append(conversation_str)
        question = "Phase 4: Thought Expand\n\nPlease think deeply about the potential solutions and generate a specific topic or question for a Google search query before expanding on the ideas."
        response = self.generate_response(question)
        self.update_memory(response)

        # Perform Google search with the generated query
        google_results = self.google_search(response)

        # Perform the secondary step (Thought Second Step)
        thought_second_step = self.thought_second_step(google_results, response)
        self.update_memory(thought_second_step)
        return thought_second_step

    def expand(self):
        conversation_str = "\n".join([f"{role_name}: {msg}" for role_name, msg in self.conversation])
        conversation_str = self.truncate_tokens(conversation_str, 14000)
        self.vector_memory.append(conversation_str)
        question = "Phase 5: Expansion\n\nPlease delve deeper into each idea and generate potential scenarios, implementation strategies, necessary partnerships or resources, and possible ways to overcome obstacles."
        response = self.generate_response(question)
        self.update_memory(response)
        return response

    def decide(self, solutions):
        conversation_str = "\n".join([f"{role_name}: {msg}" for role_name, msg in self.conversation])
        conversation_str = self.truncate_tokens(conversation_str, 14000)
        self.vector_memory.append(conversation_str)
        question = "Phase 6: Decision\n\nPlease rank the following solutions based on the evaluations, expansions, success estimation on a scale of 0 perfect to 100, and scenarios generated:\n\n"
        for i, solution in enumerate(solutions):
            question += f"{i+1}. {solution}\n"
        response = self.generate_response(question)
        self.update_memory(response)
        return response

    def produce_final_product(self, specified_task):
        conversation_str = "\n".join([f"{role_name}: {msg}" for role_name, msg in self.conversation])
        conversation_str = self.truncate_tokens(conversation_str, 14000)
        self.vector_memory.append(conversation_str)
        question = "Phase 7: Produce Final Product\n\nPlease produce/execute/code the evaluation with the highest success estimation while keeping in mind the overall task of this conversation:\n\n" + specified_task
        response = self.generate_response(question)
        self.update_memory(response)
        return response

    
    def thought_second_step(self, google_results, search_query):
        if not google_results:
            return "No search results found."

        thought_second_step_results = "Thought Second Step:\n\n"
        for i, result in enumerate(google_results):
            thought_second_step_results += f"Result {i+1}:\nTitle: {result['title']}\nURL: {result['link']}\nSnippet: {result['snippet']}\n\n"

        thought_second_step_results += "Please select the best fitting website by specifying the result index or provide any other instructions for the search."

        response = self.generate_response(thought_second_step_results)
        self.update_memory(response)

        selected_website = None
        instructions = None

        if response.isdigit():
            index = int(response) - 1
            if index >= 0 and index < len(google_results):
                selected_website = google_results[index]
        else:
            instructions = response

        if selected_website:
            search_url = selected_website["link"]
            search_results = self.perform_website_search(search_url, search_query)
            if search_results:
                search_response = "Search Results:\n\n"
                for i, result in enumerate(search_results):
                    search_response += f"Result {i+1}:\nTitle: {result['title']}\nURL: {result['link']}\n\n"
                result = f"Search results obtained for the selected website:\n\n{search_response}"
            else:
                result = "Search failed on the selected website."
        elif instructions:
            # Handle instructions provided by the agent
            result = "Instructions received: " + instructions
        else:
            result = "Invalid selection or instructions provided."

        return result

    
    def generate_response(self, prompt):
        max_tokens = 14000
        prompt_with_memory = '\n'.join([*self.vector_memory, prompt])
        prompt_with_memory = self.truncate_tokens(prompt_with_memory, max_tokens)  # truncate to fit within model's limit
        response = None
        while not response:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",  # Replace with the correct model identifier or name
                    messages=[{"role": "system", "content": prompt_with_memory}],
                )
            except openai.error.RateLimitError:
                print("Rate limit exceeded. Retrying in 15 seconds...")
                time.sleep(15)  # Delay for 15 seconds before retrying the request

        if response.choices:
            generated_response = response.choices[0].message.content.strip()
            self.update_memory(generated_response)
            return generated_response
        else:
            self.update_memory("No response from the model.")
            return "No response from the model."
        
    def perform_website_search(self, search_url, query):
        params = {
            "q": query,
            "location": "United States",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "api_key": self.serpapi_api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])

        return results

    def google_search(self, query):
        params = {
            "q": query,
            "api_key": self.serpapi_api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])

        if results:
            return results
        return []

    def update_memory(self, response):
        if isinstance(response, AIMessage):
            self.vector_memory.append(response.content)
        elif isinstance(response, str):
            self.vector_memory.append(response)
