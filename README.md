# Camel-Coder

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

## Setup

To set up and use Camel-Coder, follow these steps:

1. Install the required dependencies by running the following command: `pip install openai langchain`

2. Set up the necessary environment variables, including the OpenAI API key. You can export the API key as an environment variable or directly add it to the script. Make sure to keep your API key secure.

3. Specify the role-specific prompts for each agent by editing the `assistant_role_name`, `user_role_name`, and `task` variables in the CamelCoder.py script. Customize the role names and task according to your specific requirements.

   **Note**: Do not edit the hardcoded agents. Only modify the `assistant_role_name`, `user_role_name`, and `task` variables.

   The hardcoded agents in the script have predefined roles and responsibilities that are essential for the functionality of the chatbot. Modifying these hardcoded agents can lead to unexpected behavior and may disrupt the flow of the conversation.

   However, you have the flexibility to customize the role names and task description by modifying the `assistant_role_name`, `user_role_name`, and `task` variables. This allows you to adapt the chatbot to your specific use case and requirements while maintaining the integrity of the chatbot's functionality.

4. Run the CamelCoder.py script to start the chatbot. The script will facilitate a conversation between the agents and guide them towards completing the specified task.

## File Structure and Workspace

Camel-Coder creates a file structure within a workspace directory to organize the generated code and files. The file structure is based on the conversation between the agents and can be visualized as a tree-like structure.

To access the generated code and files, navigate to the specified conversation_directory. 

## Conversation Saving

Camel-Coder includes a feature that automatically saves the complete conversation to a text file for further analysis. To configure the conversation saving feature:

1. Open the CamelCoder.py script.
2. Locate the `conversation_directory` variable.
3. Set the desired directory where you want the conversation file to be saved, for example:

   ```python
   conversation_directory = "/path/to/conversation"

Replace /path/to/conversation with your desired directory.

When the conversation ends, the complete conversation will be saved to the specified directory.

Please note that the conversation file will be overwritten each time the script is run with the specified directory, so make sure to move or rename the previous conversation file if needed.
## Token Counting and Cost Estimation

Camel-Coder provides functionality to count the tokens used in the conversation and estimate the associated cost. After running the script, you will see the token count and cost estimation information in the console output. This helps you monitor your token usage and estimate the cost of using the OpenAI API.
Example Scenario

In this example scenario, the user acts as a project manager, and the assistant acts as a software developer. The task is to create a project plan for a new software application.
## License

This project is licensed under the MIT License. See the LICENSE file for more information.
