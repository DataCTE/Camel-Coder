# Camel-Coder

Camel-Coder is an advanced Python script powered by OpenAI's GPT-3.5-turbo model, designed to enable robust task-oriented chatbot conversations. It leverages the role-playing conversation capabilities of the GPT model to assign specific roles to different chat agents, guiding them towards achieving a specific task collaboratively.

## Features

- **Multiple Agents:** Camel-Coder provides support for multi-agent interactions. Each agent is assigned a unique role with specific responsibilities, adding depth and versatility to the chatbot capabilities.

- **Task-Oriented Conversation:** The chatbot is designed to steer the conversation towards the completion of a specific task, facilitating goal-oriented dialogue.

- **Role-Based Prompts:** Camel-Coder introduces role-specific prompt templates for each agent, offering instructions and guidelines that are tailored to each agent's function in the task completion process.

- **Intervention Mechanism:** The script incorporates an innovative intervention mechanism. A special monitor agent can intervene in the conversation to ensure the conversation remains within the bounds of the set objective.

- **Thoughtful Agent:** The script also features a thoughtful agent that proactively provides suggestions and guidance to drive the conversation in the right direction.

- **Coding Agent:** This agent generates functional prototypes based on the discussion in the conversation, providing a practical outcome from the task-oriented dialogue.

- **Conversation Saving:** It offers an automatic saving of the complete conversation to a text file, making it easy to review, analyze, or audit the conversation at any later stage.

- **Token Counting and Cost Estimation:** To ensure transparency and cost-effectiveness, the script counts the tokens used in a conversation and provides a corresponding cost estimate.

## Setup

1. **Dependencies:** Install the required dependencies using pip:

```pip install openai langchain```

markdown

2. **Environment Variables:** Set up the required environment variables. This primarily includes the OpenAI API key.

3. **Role-Specific Prompts:** Define role-specific prompts for each agent by tweaking the `assistant_role_name`, `user_role_name`, and `task` variables in the CamelCoder.py script.

4. **Execution:** Run the CamelCoder.py script to start the chatbot.

## File Structure and Workspace

Camel-Coder establishes a directory structure within the workspace to categorize the generated code and related files.

## Conversation Saving

To configure the automatic conversation-saving feature, follow the below steps:

1. Open the CamelCoder.py script.
2. Find the `conversation_directory` variable.
3. Replace the current path with the desired directory where you wish to save the conversation file:

`conversation_directory = "/your/desired/directory"`

bash

Replace `/your/desired/directory` with the actual path where you want the conversation file to be saved.

## Token Counting and Cost Estimation

After executing the Camel-Coder script, the console output will provide information about the token count and cost estimation.

## License

This project is licensed under the terms of a Custom GNU License.
