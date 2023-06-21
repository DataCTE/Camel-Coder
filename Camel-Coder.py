import os
import datetime
import time
from typing import List
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
import openai

os.environ["OPENAI_API_KEY"] =  'Your_Key'  # replace 'your-api-key' with your actual API key
openai.api_key = "Your_Key" # replace 'your-api-key' with your actual API key

conversation_directory = "/path/to/conversation.txt" # Change to disired Path


class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
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


assistant_role_name = "Ai expert"
user_role_name = "Project Lead"
task = "Produce a agent creating website. the website should create templates for openai agents for users to use"

TOKEN_LIMIT = 14000

word_limit = 50  # word limit for task brainstorming


#Hardcoded agents
thoughtful_role_name = "Thoughtful Agent"
monitor_role_name = "Monitor Agent"
coding_role_name = "Python Coding Expert"

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

coding_inception_prompt = (
    f"As the {coding_role_name}, your primary objective is to directly translate the ongoing discussion, ideas, and defined objectives into real, executable code. This isn't about summarizing, theorizing or outlining - this is about producing tangible code.\n\n"
    f"With your advanced programming skills, you're expected to craft a robust, maintainable, and scalable piece of software or application that aligns with the established requirements and expectations. Your final output must be functional, well-structured code demonstrating a keen understanding of the task at hand and a strong problem-solving ability.\n\n"
    f"Remember, this process is iterative and relies heavily on the feedback loop with other agents. It's essential to be open to suggestions, adapting and improving your work based on their inputs. The effectiveness of your work is directly correlated with how well it encapsulates the conversation into a working prototype or product.\n\n"
    f"As the {coding_role_name}, your success lies in your capacity to turn ideas into functioning code. Let's not just talk about code; let's write it. Now, generate the code!"
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
Do not provide general information or additional explanations beyond what is required.

Remember to end <YOUR_SOLUTION> with: Next question.

As we proceed, please also keep in mind the roles of other agents:
- {assistant_role_name}: Assist the user in completing the task by providing relevant information and following their instructions.
- {user_role_name}: Provide instructions and collaborate with other agents.
- {thoughtful_role_name}: Provide thoughtful suggestions to guide the conversation and contribute to the overall progress.
- {coding_role_name}: Develop a large and complex prototype based on the ongoing discussion.
- {monitor_role_name}: Observe the conversation and ensure that all agents are adhering to the task goal. Intervene when necessary.

Let's collaborate effectively to accomplish our task!"""
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
    Let's collaborate effectively to accomplish our task!"""
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
"""
)

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


def write_conversation_to_file(conversation, filename):
    def timestamp():
        now = datetime.datetime.now()
        timestamp = now.strftime("%H%M%d%m%Y")
        return timestamp

    def append_timestamp_to_filename(filename):
        base, extension = os.path.splitext(filename)
        new_filename = f"{base}-{timestamp()}{extension}"
        return new_filename

    filename = os.path.join(conversation_directory, append_timestamp_to_filename(filename))

    try:
        with open(filename, 'w') as f:
            for turn in conversation:
                speaker, statement = turn
                f.write(f"{speaker}: {statement}\n\n")
        print(f"Conversation successfully written to {filename}")
    except Exception as e:
        print(f"Failed to write conversation to file: {e}")

    filename = append_timestamp_to_filename(filename)

    with open(filename, 'w') as f:
        for turn in conversation:
            speaker, statement = turn
            f.write(f"{speaker}: {statement}\n\n")


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

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)

conversation = []
total_tokens = 0
preserve_last_complete_message = True

assistant_inception_msg = SystemMessage(content=assistant_inception_prompt)
thoughtful_inception_msg = SystemMessage(content=thoughtful_inception_prompt)
coding_inception_msg = SystemMessage(content=coding_inception_prompt)
user_inception_msg = SystemMessage(content=user_inception_prompt)
user_agent.init_messages()
user_agent.update_messages(user_inception_msg)

# Initialize the MonitorAgent
monitor_agent = MonitorAgent(monitor_inception_prompt, "gpt-3.5-turbo-16k")

# Add agents to the list
agents = [
    (user_role_name, user_agent, HumanMessage, user_inception_msg),
    (assistant_role_name, assistant_agent, HumanMessage, assistant_inception_msg),
    (thoughtful_role_name, thoughtful_agent, AIMessage, thoughtful_inception_msg),
    (coding_role_name, coding_agent, AIMessage, coding_inception_msg),
]

# Set the number of loops for user, assistant, and thoughtful agents
loop_count = 2

# Set the number of main loops before running the coding agent and monitor agent intervention
main_loops_before_coding = 2
main_loops_before_monitor_intervention = 5

# Main conversation loop
with get_openai_callback() as cb:
    chat_turn_limit, n = 50, 0
    main_loop_count = 0

    while n < chat_turn_limit:
        n += 1
        separator_line = "\n" + "=" * 60 + "\n"

        
     # User, Assistant, Thoughtful loop
    for _ in range(loop_count):
        for i, (role_name, agent, MessageClass, inception_msg) in enumerate(agents[:-1]):
            if n == 1 and role_name == user_role_name:
                ai_msg = agent.step(inception_msg)
            else:
                # Gather previous agent messages excluding the current agent's own responses
                prev_agent_responses = [msg for agent_name, msg in conversation if agent_name != role_name]
                message_content = "\n".join(prev_agent_responses[-2:])  # use only the two most recent responses
                ai_msg = agent.step(AIMessage(content=message_content))

            msg = MessageClass(content=ai_msg.content)
            conversation.append((role_name, msg.content))
            total_tokens += len(msg.content.split())
            print(separator_line)
            print(f"\n{'-' * 50}\n{role_name}:\n{'-' * 50}\n{msg.content}\n")
            print(separator_line)

            if total_tokens > TOKEN_LIMIT:
                print("Token limit exceeded. Truncating conversation.")
                if preserve_last_complete_message:
                    last_complete_message = "\n".join([msg.content for _, _, msg, _ in agents[i-1:i-2]])

        # Increment the main_loop_count after one full loop
        main_loop_count += 1

        # Coding agent loop after main_loops_before_coding full main loops
        if main_loop_count % main_loops_before_coding == 0:
            for _ in range(loop_count):
                role_name, coding_agent, MessageClass, coding_inception_msg = agents[-1]
                recent_loop = "\n".join([msg.content for _, _, msg, _ in agents[:-1] if isinstance(msg, AIMessage)])
                if total_tokens + len(recent_loop.split()) > TOKEN_LIMIT:
                    print("Token limit exceeded. Skipping Coding agent's response.")
                    continue
                recent_loop = truncate_text(recent_loop, TOKEN_LIMIT - total_tokens)
                coding_inception_msg = SystemMessage(content=coding_inception_prompt)
                coding_ai_msg = coding_agent.step(AIMessage(content=recent_loop))
                coding_msg = AIMessage(content=coding_ai_msg.content)
                conversation.append((role_name, coding_msg.content))
                total_tokens += len(coding_msg.content.split())
                print(separator_line)
                print(f"\n{'-' * 50}\n{role_name}:\n{'-' * 50}\n{coding_msg.content}\n")
                print(separator_line)

                # Prompt the coding agent to refine the product
                refinement_prompt = f"As the {coding_role_name}, Now provide a codebase that conforms to the highlevel outline you just produced. You MUST MUST MUST MUST PRODUCE THE CODE FOR THIS OUTLINE!!!"
                refinement_ai_msg = coding_agent.step(SystemMessage(content=refinement_prompt))
                refinement_msg = AIMessage(content=refinement_ai_msg.content)
                conversation.append((role_name, refinement_msg.content))
                total_tokens += len(refinement_msg.content.split())
                print(separator_line)
                print(f"\n{'-' * 50}\n{role_name} (Refinement):\n{'-' * 50}\n{refinement_msg.content}\n")
                print(separator_line)

                # Feed the refined response to the user agent
                user_agent.step(AIMessage(content=refinement_msg.content))

                # Increment the main_loop_count after the coding agent's response
                main_loop_count += 1

        # Monitor agent intervention after main_loops_before_monitor_intervention full main loops
        if main_loop_count % main_loops_before_monitor_intervention == 0:
            monitor_msg_content = monitor_agent.stage_intervention(conversation)
            print(separator_line)
            print(f"\n{'-' * 50}\nMonitor Intervention:\n{'-' * 50}\n{monitor_msg_content}\n")
            print(separator_line)

            # Feed the intervention message to all agents
            for role_name, agent, _, _ in agents:
                agent.step(SystemMessage(content=monitor_msg_content))

        if any("<TASK_DONE>" in msg.content for _, _, msg, _ in agents if isinstance(msg, AIMessage)):
            break

        time.sleep(1)

    print(f"Total Successful Requests: {cb.successful_requests}")
    print(f"Total Tokens Used: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

write_conversation_to_file(conversation, 'conversation.txt')
