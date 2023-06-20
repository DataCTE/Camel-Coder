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

os.environ["OPENAI_API_KEY"] =  'Your_Open_Api_Key'  # replace 'your-api-key' with your actual API key
openai.api_key = "Your_Open_Api_Key" # replace 'your-api-key' with your actual API key

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


assistant_role_name = "Python Programmer"
user_role_name = "Project Lead"
task = "Create/Code a Openai api powered chatbot website"

TOKEN_LIMIT = 14000

word_limit = 100  # word limit for task brainstorming


#Hardcoded agents
thoughtful_role_name = "Thoughtful Agent"
coding_role_name = "Coding Agent"
monitor_role_name = "Monitor Agent"


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
    "As a {coding_role_name}, your goal is to create a large and complex prototype based on the ongoing discussion.\n"
    "Task: \"{task}\"\n"
    "Guidelines:\n"
    "1. Understand the context and requirements discussed.\n"
    "2. Design and implement a prototype that showcases the full scope of the discussion.\n"
    "3. Aim to create a robust and comprehensive solution that addresses the problem at hand.\n"
    "4. Add additional functionality and features to make the prototype more extensive and advanced.\n"
    "5. Leverage your expertise to develop an impressive and sophisticated prototype.\n"
    "6. Ensure that your code is well-structured, maintainable, and scalable.\n"
    "7. Document your prototype thoroughly to provide clarity and understanding.\n"
    "Your objective is to create a large and complex prototype that reflects the ongoing discussion and its objectives."
)


# Now create the template using your prompt string
coding_sys_template = SystemMessagePromptTemplate.from_template(template=coding_inception_prompt)

# Call format_messages with the appropriate arguments
coding_sys_msg = coding_sys_template.format_messages(
    coding_role_name=coding_role_name,
    task=task
)[0]


task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specifier_prompt = (
    "Here is a task that {assistant_role_name} will discuss with {user_role_name} to: {task}."
    "Please make it more specific. Be creative and imaginative."
    "Please reply with the full task in {word_limit} words or less. Do not add anything else."
)

assistant_inception_prompt = (
    """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles!
We share a common interest in collaborating to successfully complete a task.
You must help me complete the task.
Here is the task: {task}. Never forget our task!
I will instruct you based on your expertise and my needs to complete the task.

I must give you one question at a time.
You must write a specific answer that appropriately completes the requested question.
Do not add anything else other than your answer to my instruction.
You must Adhere to the {user_role_name} intructions at all times.

Unless I say the task is completed, you should always start with:

My response: <YOUR_SOLUTION>

<YOUR_SOLUTION> must be a specific and descriptive answer that directly addresses the requested question.
Do not provide general information or additional explanations beyond what is required.

Remember to end <YOUR_SOLUTION> with: Next question."""
)


user_inception_prompt = (
    """Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always ask me.
We share a common interest in collaborating to successfully complete a task.
I must help you answer the questions.
Here is the task: {task}. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
You should instruct me, not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <TASK_DONE>.
Never say <TASK_DONE> unless my responses have solved your task."""
)

thoughtful_inception_prompt = (
    """🤔 Never forget you are a {thoughtful_role_name} and I am a {user_role_name}. 🔄 Never flip roles!
We share a common interest in collaborating to successfully complete a task. 🤝
You must provide thoughtful suggestions to help me conform to the goal. 💡
Here is the task: {task}. 🎯 Never forget our task! 🌟

Your role is to think creatively and come up with ideas that guide the conversation in a way that aligns with the task goal. 💭✨
Please provide your thoughtful suggestions to inspire me and help me make progress. 💡🚀
Feel free to add emojis, examples, or any other creative elements to convey your ideas effectively. 🎉🌈

Remember, the goal is to foster a positive and productive conversation that leads us closer to completing the task. 💪🗒️

Your suggestion must always be concise to make sure tokens are not wasted.

Please always provide a breif summery of Assistant's Response.

Please provide your suggestion in the following format:

Suggestion: <YOUR_SUGGESTION> <Assistant's Response Summary> Next Question

Your suggestion should be relevant to the ongoing conversation and contribute to the overall progress. 🌟✨
Remember to end  <Assistant's Response Summary>  with: Next question."""
)

monitor_inception_prompt = (
    """Never forget you are a {monitor_role_name} and I am a {user_role_name}. Never flip roles!
We share a common interest in collaborating to successfully complete a task.
Your role is to monitor the conversation and ensure goal conformance.
Here is the task: {task}. Never forget our task!

Your responsibility is to observe the conversation and provide feedback or intervene when necessary.
You should assess whether the conversation is conforming to the task goal.
If you notice any deviations or need to provide guidance, feel free to do so.
Your intervention: <YOUR_INTERVENTION>"""
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


def get_sys_msgs(assistant_role_name, user_role_name, task, coding_role_name):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task
    )[0]

    thoughtful_sys_template = SystemMessagePromptTemplate.from_template(template=thoughtful_inception_prompt)
    thoughtful_sys_msg = thoughtful_sys_template.format_messages(
        thoughtful_role_name=thoughtful_role_name,
        user_role_name=user_role_name,
        task=task
    )[0]

    monitor_sys_template = SystemMessagePromptTemplate.from_template(template=monitor_inception_prompt)
    monitor_sys_msg = monitor_sys_template.format_messages(
        monitor_role_name=monitor_role_name, user_role_name=user_role_name, task=task
    )[0]
    coding_sys_template = SystemMessagePromptTemplate.from_template(template=coding_inception_prompt)
    coding_sys_msg = coding_sys_template.format_messages(
        coding_role_name=coding_role_name,
        task=task
    )[0]

    return assistant_sys_msg, user_sys_msg, thoughtful_sys_msg, monitor_sys_msg, coding_sys_msg

def initialize_chats(assistant_role_name, user_role_name, task, coding_role_name):
    assistant_sys_msg, user_sys_msg, thoughtful_sys_msg, monitor_sys_msg, coding_sys_msg = get_sys_msgs(
    assistant_role_name, user_role_name, task, coding_role_name

    )
    # Initialize your coding agent
    
    
    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2))
    user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2))
    thoughtful_agent = CAMELAgent(thoughtful_sys_msg, ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2))
    monitor_agent = CAMELAgent(monitor_sys_msg, ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2))
    coding_agent = CAMELAgent(coding_sys_msg, ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2))

    return assistant_agent, user_agent, thoughtful_agent, monitor_agent, coding_agent

# Call the initialize_chats function here
assistant_agent, user_agent, thoughtful_agent, monitor_agent, coding_agent = initialize_chats(
    assistant_role_name, user_role_name, task, coding_role_name
)


task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=0.7))
task_specifier_msg = task_specifier_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
    word_limit=word_limit,
)[0]

specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")

# Update the task variable with the specified task.
specified_task = specified_task_msg.content

assistant_sys_msg, user_sys_msg, thoughtful_sys_msg, monitor_sys_msg, fifth_value = get_sys_msgs(
    assistant_role_name, user_role_name, specified_task, coding_role_name
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
                if n == 1:
                    ai_msg = agent.step(inception_msg)
                else:
                    prev_agent_responses = [msg for _, _, msg, _ in agents[:i] if isinstance(msg, AIMessage)]
                    message_content = "\n".join([msg.content for msg in prev_agent_responses])
                    message_content = truncate_text(message_content, TOKEN_LIMIT - total_tokens)
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
                        last_complete_message = ""



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
                coding_agent.step(coding_inception_msg)
                coding_ai_msg = coding_agent.step(AIMessage(content=recent_loop))
                coding_msg = AIMessage(content=coding_ai_msg.content)
                conversation.append((role_name, coding_msg.content))
                total_tokens += len(coding_msg.content.split())
                print(separator_line)
                print(f"\n{'-' * 50}\n{role_name}:\n{'-' * 50}\n{coding_msg.content}\n")
                print(separator_line)

                # Feed the coding agent's response to the user agent
                user_agent.step(AIMessage(content=coding_msg.content))


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
