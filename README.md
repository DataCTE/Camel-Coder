## Camel-Coder

Camel-Coder is a Python script that implements a chatbot using OpenAI's GPT-3.5-turbo model. The chatbot facilitates a conversation between multiple agents, each playing a specific role, to collaboratively complete a given task. This script is an iteration of the previous project Camel-local.

## Features

- Multiple agents: The chatbot supports multiple agents, each with their own role and responsibilities.
- Task-oriented conversation: The chatbot guides the conversation towards completing a specific task.
- Role-based prompts: Each agent has its own system message prompt template, providing role-specific instructions and guidelines.
- Intervention mechanism: The chatbot includes a monitor agent that can intervene in the conversation if necessary to ensure goal conformance.
- Thoughtful agent: The chatbot includes a thoughtful agent that provides suggestions to guide the conversation in a productive direction.
- Coding agent: The chatbot includes a coding agent that generates a functional prototype based on the ongoing discussion.
- Conversation saving: The complete conversation can be automatically saved to a text file for further analysis.
- Token counting and cost estimation: The script provides functionality to count the tokens used in the conversation and estimate the associated cost.
- Modular design: The script has been refactored into multiple modules, each responsible for a specific aspect of the chatbot's functionality. This makes the script easier to maintain, extend, and test.
  
## Setup

To set up and use Camel-Coder, follow these steps:

1. Install the required dependencies by running the following command:
``` pip install openai langchain ```

2. Set up the necessary environment variables in `config.py`, including the OpenAI API key. You can export the API key as an environment variable or directly add it to the script. Make sure to keep your API key secure.

3. Specify the role-specific prompts for each agent by editing the assistant_role_name, user_role_name, and task variables in the CamelCoder.py script. Customize the role names and task according to your specific requirements.
  
## Note ##: Do not edit the hardcoded agents. Only modify the assistant_role_name, user_role_name, and task variables.

The hardcoded agents in the script have predefined roles and responsibilities that are essential for the functionality of the chatbot. Modifying these hardcoded agents can lead to unexpected behavior and may disrupt the flow of the conversation.

However, you have the flexibility to customize the role names and task description by modifying the assistant_role_name, user_role_name, and task variables. This allows you to adapt the chatbot to your specific use case and requirements while maintaining the integrity of the chatbot's functionality.

## Conversation Saving

Camel-Coder includes a feature that automatically saves the complete conversation to a text file for further analysis. To configure the conversation saving feature:

1. Open the CamelCoder.py script.
2. Locate the `conversation_file_path` variable.
3. Set the desired file path where you want the conversation file to be saved, for example:

```
conversation_directory = "/path/to/conversation.txt"
```
Replace /path/to/conversation.txt with your desired file path.

When the conversation ends, the complete conversation will be saved to the specified file path.

Please note that the conversation file will be overwritten each time the script is run with the specified file path, so make sure to move or rename the previous conversation file if needed.

## Token Counting and Cost Estimation

Camel-Coder provides functionality to count the tokens used in the conversation and estimate the associated cost. After running the script, you will see the token count and cost estimation information in the console output. This helps you monitor your token usage and estimate the cost of using the OpenAI API.

Requirements

    Python 3.7 or higher
    OpenAI Python library
    Langchain library

## Note

This version of the chatbot utilizes the OpenAI API for easier user setup

## License


This project is licensed under the terms of a Custom GNU License. See the LICENSE file for more information.
