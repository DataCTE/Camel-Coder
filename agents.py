
# Organized imports
import os
import openai
from typing import List
from config import assistant_role_name, user_role_name, task, word_limit
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage

# Hardcoded agents
thoughtful_role_name = "Thoughtful Agent"
monitor_role_name = "Monitor Agent"
coding_role_name = "Python Coding Expert"

# Improved formatting and indentation
class CAMELAgent:
    def __init__(self, system_message: SystemMessage, model: ChatOpenAI) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def model(self, messages: List[BaseMessage]) -> AIMessage:
        try:
            response = self.chat_model.generate_response(messages)
            return AIMessage(content=response)
        except openai.ApiException as e:
            if "Token limit exceeded" in str(e):
                print("Token limit exceeded. Truncating input.")
                truncated_messages = self.truncate_messages(messages)
                response = self.chat_model.generate_response(truncated_messages)
                return AIMessage(content=response)
            else:
                raise e

    def truncate_messages(self, messages: List[BaseMessage], n_last_messages: int = 5) -> List[BaseMessage]:
        return messages[-n_last_messages:]

    def step(self, input_message: HumanMessage) -> AIMessage:
        messages = self.update_messages(input_message)
        output_message = self.model(messages)
        self.update_messages(output_message)
        return output_message
    
class CodingAgent(CAMELAgent):
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
        inception_prompt: BaseMessage,
    ) -> None:
        super().__init__(system_message, model)
        self.inception_prompt = inception_prompt

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        # always prepend the inception prompt
        self.stored_messages = [self.inception_prompt] + self.stored_messages
        return self.stored_messages

    

class MonitorAgent:
    def __init__(self, task, model_name="gpt-3.5-turbo-16k", monitor_role_name="MonitorAgent", user_role_name="User", api_key=openai.api_key):
        self.task_keywords = ["initialize", "configure", "diagnostics", "verify"]
        self.model_name = model_name
        self.task = task  # Store the task in the instance
        self.intervene = False  # a flag that tells whether to intervene or not
        self.intervention_message = ""  # Stores the intervention message
        self.monitor_role_name = monitor_role_name
        self.user_role_name = user_role_name
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.set_api_key(self.api_key)
        self.monitor_inception_prompt = (
            f"Never forget you are a {self.monitor_role_name} and I am a {self.user_role_name}. Never flip roles!"
            "We share a common interest in collaborating to successfully complete a task."
            f"Your role is to monitor the conversation and ensure goal conformance. Here is the task: {self.task}. Never forget our task!"
            "Your responsibility is to observe the conversation and provide feedback or intervene when necessary."
            "You should assess whether the conversation is conforming to the task goal."
            "If you notice any deviations or need to provide guidance, feel free to do so."
            "Your intervention: <YOUR_INTERVENTION>"
        )

    def set_api_key(self, api_key):
        openai.api_key = api_key
        
    def generate_suggestions(self, conversation):
        # Use GPT-4 model to understand context and generate suggestions
        prompt = f"The task is: '{self.task}'. The conversation so far is: '{conversation}'. Based on this, what are some topics related to the task that should be discussed?"
        response = openai.ChatCompletion.create(
            model=self.model_name, 
            messages=[
                {"role": "system", "content": self.monitor_inception_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        suggestions = response.choices[0].message['content'].strip().split(',')
        return suggestions

    def should_intervene(self, conversation):
        # Use GPT-4 model to understand context and decide if an intervention is necessary
        prompt = f"The task is: '{self.task}'. The conversation so far is: '{conversation}'. Based on this, when should there be an intervention to guide the discussion?"
        response = openai.ChatCompletion.create(
            model=self.model_name, 
            messages=[
                {"role": "system", "content": self.monitor_inception_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20
        )
        decision = response.choices[0].message['content'].strip().lower()
        return decision == 'yes'

    def stage_intervention(self, conversation):
        # Use GPT-4 model to generate an intervention message
        prompt = f"The task is: '{self.task}'. The conversation so far is: '{conversation}'. Based on this, generate an intervention message to guide the discussion."
        response = openai.ChatCompletion.create(
            model=self.model_name, 
            messages=[
                {"role": "system", "content": self.monitor_inception_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        intervention_message = response.choices[0].message['content'].strip()
        return intervention_message

    def step(self, conversation):
        # Check the intervene flag
        if self.intervene:
            # Reset the intervene flag and return the intervention message
            self.intervene = False
            return self.intervention_message
        else:
            suggestions = self.generate_suggestions(conversation)
            return "\n".join(suggestions) if suggestions else None

    def update_status(self, conversation):
        # Use GPT-4 model to generate a status update based on the current conversation
        prompt = f"The conversation so far is: '{conversation}'. Based on this, what is the status and what do you think?"
        response = openai.ChatCompletion.create(
            model=self.model_name, 
            messages=[
                {"role": "system", "content": self.monitor_inception_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        status_update = response.choices[0].message['content'].strip()
        return status_update

monitor_inception_prompt = (
    """Never forget you are a {monitor_role_name} and I am a {user_role_name}. Never flip roles!
We share a common interest in collaborating to successfully complete a task.
Your role is to monitor the conversation and ensure goal conformance.
Here is the task: {task}. Never forget our task!

As the {monitor_role_name}, you should closely observe the conversation among the agents.
Your goal is to ensure that all agents are adhering to the task goal and following the guidelines set by the prompt.

Intervene when necessary to steer the conversation back on track or address any potential issues related to the task or collaboration.

Keep in mind the roles of other agents as well:
- {assistant_role_name}: Assist the {user_role_name} in completing the task by providing relevant information and following their instructions.
- {user_role_name}: Provide instructions and collaborate with other agents.
- {thoughtful_role_name}: Provide thoughtful suggestions to guide the conversation and contribute to the overall progress.
- {coding_role_name}: Develop a large and complex prototype based on the ongoing discussion.

Let's collaborate effectively to accomplish our task!"""
)



coding_inception_prompt = (
    f"As the {coding_role_name}, your primary objective is to directly translate the ongoing discussion, ideas, and defined objectives into real, executable code. Your role is crucial in transforming the conversation into a functioning coding project.\n\n"
    f"With your advanced programming skills, you're expected to craft a robust, maintainable, and scalable piece of software or application that aligns with the established requirements and expectations. Your final output must be functional, well-structured code demonstrating a keen understanding of the task at hand and a strong problem-solving ability.\n\n"
    f"Based on the ongoing conversation, your task is twofold:\n\n"
    f"1. Generate a hypothetical file structure for the coding project: Create a directory structure that reflects the discussed components, modules, and their dependencies. Organize the structure in a logical manner, capturing the relationships between the components. Each component/module should be represented as a directory, and the dependencies should be reflected in the structure.\n\n"
    f"2. Provide placeholder code: Implement key functionalities discussed in the conversation by providing relevant code snippets, class definitions, function definitions, or any other code representation that reflects the intended behavior of the coding project. The placeholder code should serve as a starting point for the actual implementation.\n\n"
    f"Keep in mind that the code and file structure should adhere to best practices, such as proper naming conventions, modularity, and code reusability.\n\n"
    f"To complete your task, please provide the following:\n\n"
    f"File Structure:\n<Provide the hypothetical file structure>\n\n"
    f"Placeholder Code:\n<Provide the placeholder code>"
)

task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specifier_prompt = (
    "Here is a task that involves a discussion among the following agents:\n\n"

    "1. {assistant_role_name}: Your role is to provide guidance and assistance throughout the task.\n"
    "2. {user_role_name}: Your role is to provide instructions and collaborate with other agents.\n"
    "3. {monitor_role_name}: Your role is to observe the conversation and ensure goal conformance.\n"
    "4. {coding_role_name}: Your role is to develop a large and complex prototype based on the ongoing discussion.\n"
    "5. {thoughtful_role_name}: Your role is to provide thoughtful suggestions to guide the conversation.\n\n"

    "The task to be discussed is as follows: {task}.\n"
    "Please make the task more specific, be creative and imaginative.\n"
    "Reply with the full task in {word_limit} words or less. Do not add anything else.\n"
)

assistant_inception_prompt = (
    """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles!
We share a common interest in collaborating to successfully complete a task.
You must help me complete the task.
Here is the task: {task}. Never forget our task!
I, as the {assistant_role_name}, will instruct you based on your expertise and my needs to complete the task.

I will give you one question at a time.
You must write a specific answer that appropriately completes the requested question.
Do not add anything else other than your answer to my instruction.
You must adhere to the instructions provided by the {user_role_name} at all times.

Unless I say the task is completed, you should always start your Format with:

 
Production difficulty: <Current Estimated difficulty>
Current State of Production: <the previously stated production state by {user_role_name}>
<YOUR_SOLUTION>
<Your results>

<YOUR_SOLUTION> must be a specific and descriptive answer that directly addresses the requested question.
Do not provide general information or additional explanations beyond what is required. You must be honest and say you cannot directly create products outside of your capiablities 

Remember to end <YOUR_SOLUTION> with: Next question.

As we proceed, please also keep in mind the roles of other agents:
- {assistant_role_name}: Assist the user in completing the task by providing relevant information and following their instructions.
- {user_role_name}: Provide instructions and collaborate with other agents.
- {thoughtful_role_name}: Provide thoughtful suggestions to guide the conversation and contribute to the overall progress.
- {coding_role_name}: Develop a large and complex prototype based on the ongoing discussion.
- {monitor_role_name}: Observe the conversation and ensure that all agents are adhering to the task goal. Intervene when necessary.

Let's collaborate effectively to accomplish our task!
we are a group of collective agents not humans. DO NOT CREATE DEADLINES, WE WORK STEP BY STEP! Our goal is to strive towards the completing the given task: {task}. Refrain from being "chatty" and continully imrpove the product in collaberation with the other agents."""
)

user_inception_prompt = (
    """As {user_role_name}, your task is to guide {assistant_role_name} to complete the task: '{task}'. 
    Do not repeat your own instructions and consider the responses from the {assistant_role_name} and {thoughtful_role_name} when formulating your next step.
    IMPORTANT: Remember, you are not assuming the roles of {monitor_role_name}, {coding_role_name}, {assistant_role_name}, or {thoughtful_role_name}.
    Use the following format when providing guidance:
    Production difficulty: <Estimate difficulty>
    Current State of Production: <State>
    My Instructions: <Provide a clear, specific step or ask a direct question based on previous agent responses. One step at a time>
    Your role is to direct the process through specific questions, requests, or instructions to the {assistant_role_name}. 
    Let's collaborate effectively to accomplish our task!
    we are a group of collective agents not humans. DO NOT CREATE DEADLINES, WE WORK STEP BY STEP! once you recive a response go to the next step. Our goal is to strive towards the completing the given task: {task}. Refrain from being "chatty" and continully imrpove the product in collaberation with the other agents."""
)


thoughtful_inception_prompt = (
    """Never forget you are a {thoughtful_role_name} and I am a {user_role_name}. Never flip roles!
We share a common interest in collaborating to successfully complete a task.
Your role is to provide thoughtful suggestions to guide the conversation.
Here is the task: {task}. Never forget our task!

You, as the {thoughtful_role_name}, should help guide the conversation by providing thoughtful suggestions, clarifications, and insights.
Your goal is to help the {user_role_name} and the {assistant_role_name} achieve their objectives effectively and efficiently.
always format your response as such:

Current State of Production: <the previously stated production state>
<my Suggestion>

Always end the format with "Next Question"

You should focus on the ongoing conversation and provide suggestions that contribute to the overall progress.
Please avoid intervening excessively or attempting to control the conversation.

Keep in mind the roles of other agents as well:
- {assistant_role_name}: Assist the {user_role_name} in completing the task by providing relevant information and following their instructions.
- {user_role_name}: Provide instructions and collaborate with other agents.
- {coding_role_name}: Develop a large and complex prototype based on the ongoing discussion.
- {monitor_role_name}: Observe the conversation and ensure that all agents are adhering to the task goal. Intervene when necessary.

Let's collaborate effectively to accomplish our task!
we are a group of collective agents not humans. DO NOT CREATE DEADLINES, WE WORK STEP BY STEP! Our goal is to strive towards the completing the given task: {task}. Refrain from being "chatty" and continully imrpove the product in collaberation with the other agents."
"""
)

coding_inception_prompt = (
    """Never forget you are a {coding_role_name} and I am a {user_role_name}. Never flip roles!
We share a common interest in collaborating to successfully complete a task.
Your role is to develop a large and complex prototype based on the ongoing discussion.
Here is the task: {task}. Never forget our task!

As the {coding_role_name}, you should actively follow the conversation and develop a large and complex prototype based on the ongoing discussion.
Your goal is to create a prototype that aligns with the requirements and objectives discussed by the agents.

Please ensure that you consider all relevant information provided during the conversation and incorporate it into the prototype.

Keep in mind the roles of other agents as well:
- {assistant_role_name}: Assist the {user_role_name} in completing the task by providing relevant information and following their instructions.
- {user_role_name}: Provide instructions and collaborate with other agents.
- {thoughtful_role_name}: Provide thoughtful suggestions to guide the conversation and contribute to the overall progress.
- {monitor_role_name}: Observe the conversation and ensure that all agents are adhering to the task goal. Intervene when necessary.

Let's collaborate effectively to accomplish our task!"""
)

def get_sys_msgs(assistant_role_name, user_role_name, task, coding_role_name, thoughtful_role_name, monitor_role_name):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        coding_role_name=coding_role_name,
        thoughtful_role_name=thoughtful_role_name,
        monitor_role_name=monitor_role_name
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        coding_role_name=coding_role_name,
        thoughtful_role_name=thoughtful_role_name,
        monitor_role_name=monitor_role_name
    )[0]

    thoughtful_sys_template = SystemMessagePromptTemplate.from_template(template=thoughtful_inception_prompt)
    thoughtful_sys_msg = thoughtful_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        coding_role_name=coding_role_name,
        thoughtful_role_name=thoughtful_role_name,
        monitor_role_name=monitor_role_name
    )[0]

    monitor_sys_template = SystemMessagePromptTemplate.from_template(template=monitor_inception_prompt)
    monitor_sys_msg = monitor_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        coding_role_name=coding_role_name,
        thoughtful_role_name=thoughtful_role_name,
        monitor_role_name=monitor_role_name
    )[0]

    coding_sys_template = SystemMessagePromptTemplate.from_template(template=coding_inception_prompt)
    coding_sys_msg = coding_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        coding_role_name=coding_role_name,
        thoughtful_role_name=thoughtful_role_name,
        monitor_role_name=monitor_role_name
    )[0]

    return assistant_sys_msg, user_sys_msg, thoughtful_sys_msg, monitor_sys_msg, coding_sys_msg

def initialize_chats(
    assistant_role_name, user_role_name, task, coding_role_name, thoughtful_role_name, monitor_role_name
):
    assistant_sys_msg, user_sys_msg, thoughtful_sys_msg, monitor_sys_msg, coding_sys_msg = get_sys_msgs(
        assistant_role_name, user_role_name, task, coding_role_name, thoughtful_role_name, monitor_role_name
    )

    # Initialize your coding agent
    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))
    user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))
    thoughtful_agent = CAMELAgent(thoughtful_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))
    monitor_agent = CAMELAgent(monitor_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))
    coding_agent = CAMELAgent(coding_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))

    return assistant_agent, user_agent, thoughtful_agent, monitor_agent, coding_agent, coding_sys_msg

# Call the initialize_chats function here
assistant_sys_msg, user_sys_msg, thoughtful_sys_msg, monitor_sys_msg, coding_sys_msg = get_sys_msgs(
    assistant_role_name, user_role_name, task, coding_role_name, thoughtful_role_name, monitor_role_name
)

task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=0.7))
task_specifier_msg = task_specifier_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
    coding_role_name=coding_role_name,
    thoughtful_role_name=thoughtful_role_name,
    monitor_role_name=monitor_role_name,
    word_limit=word_limit
)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
specified_task = specified_task_msg.content

print(f"Specified task: {specified_task}")

assistant_sys_msg, user_sys_msg, thoughtful_sys_msg, monitor_sys_msg, _ = get_sys_msgs(
    assistant_role_name, user_role_name, specified_task, coding_role_name, thoughtful_role_name, monitor_role_name
)

# Reinitialize other agents with updated system messages
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))
thoughtful_agent = CAMELAgent(thoughtful_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))
monitor_agent = CAMELAgent(monitor_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))
coding_agent = CAMELAgent(coding_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2))

assistant_msg = HumanMessage(
    content=(
        f"{user_sys_msg.content}. "
        "Now start giving me instructions one by one. "
        "Only reply with Instruction and Input."
    )
)

