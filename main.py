from file_operations import write_conversation_to_file, generate_file_structure_and_scripts
from file_interactor import FileInteractor
from agents import (
    specified_task_msg, assistant_inception_prompt, thoughtful_inception_prompt, coding_inception_prompt,
    user_role_name, assistant_role_name, thoughtful_role_name, coding_role_name,
    user_agent, user_inception_prompt, MonitorAgent, assistant_agent, coding_agent, thoughtful_agent,
    specified_task
)
from langchain.prompts.chat import SystemMessage, HumanMessage, AIMessage
from config import TOKEN_LIMIT, task, serpapi_api_key, folder_path
from tot import ProblemSolver
import openai

# Initialize your FileInteractor
file_interactor = FileInteractor(folder_path)

# Initialize conversation
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
monitor_agent_class = MonitorAgent(
    """The goal of the agents in this program is to continually improve the existing codebase 
    while preserving existing functionality. This means that any updates or modifications made 
    to the agents' operations should aim to increase efficiency, readability, or capabilities 
    without negatively affecting the outcomes they currently produce.""",
    "gpt-3.5-turbo-16k"
)

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

problem_solver = ProblemSolver(
    openai_key=openai.api_key,
    serpapi_api_key=serpapi_api_key
)

# Initialize codebase content list
codebase_content = []

# Get the files from the folder
files = file_interactor.get_files()

separator_line = "\n" + "=" * 60 + "\n"

# Iterate over each file
for file_index, file_path in enumerate(files):
    print(f"Processing file: {file_path}")

    # Get the lines from the file
    lines = file_interactor.get_lines(file_path)

    # Join the lines to get the file content
    file_content = "\n".join(lines)

    # Append the file content to the codebase content list
    codebase_content.append(file_content)

    # After you've received the response from the coding agent
    updated_file_content = file_content

    # Write the updated file content to the file
    file_interactor.save_file(file_path, updated_file_content)

    # Initialize conversation for the current file
    conversation = []
    user_agent.init_messages()
    user_agent.update_messages(user_inception_msg)

    # Set the goal for the ProblemSolver at the start of the conversation loop
    problem_solver.set_goal(
        """The goal of the agents in this program is to continually improve the existing codebase 
        while preserving existing functionality. This means that any updates or modifications made 
        to the agents' operations should aim to increase efficiency, readability, or capabilities 
        without negatively affecting the outcomes they currently produce.""",
        conversation
    )
    
    # Main conversation loop
    chat_turn_limit = 50
    main_loop_count = 0

    for n in range(chat_turn_limit):
        separator_line = "\n" + "=" * 60 + "\n"

         # User, Assistant, Thoughtful loop
        for _ in range(loop_count):
            for i, (role_name, agent, MessageClass, inception_msg) in enumerate(agents[:-1]):
                # Read the file content
                file_content = file_interactor.read_file(file_path)

                # Gather previous agent messages excluding the current agent's own responses
                prev_agent_responses = [msg[1] for msg in conversation if msg[0] != role_name]

                # Combine the file content with the previous agent responses (excluding the current agent's own response)
                all_messages = [file_content] + prev_agent_responses

                # Filter out messages that are not strings or AIMessage objects
                all_messages = [msg for msg in all_messages if isinstance(msg, str) or isinstance(msg, AIMessage)]

                # Extract the content from each message and join them with a newline
                message_content = "\n".join(
                    [msg.content if isinstance(msg, AIMessage) else msg for msg in all_messages])

                # Feed the previous agent responses and file content into the agent's step function
                ai_msg = agent.step(AIMessage(content=message_content))

                # Print agent message to screen
                print(separator_line)
                print(f"\n{'-' * 50}\n{role_name} Message:\n{'-' * 50}\n{ai_msg.content}\n")
                print(separator_line)
                # Add agent message to conversation history
                conversation.append((role_name, ai_msg))

                # Update the file content after the coding agent's loop
                if main_loop_count % main_loops_before_coding == 0:
                    # Read the file content
                    file_content = file_interactor.read_file(file_path)

                    # Generate the updated file content based on the codebase content
                    updated_file_content = "\n".join(codebase_content)

                    # Verify if the updated file content matches the expected content
                    if not file_interactor.verify_file_content(file_path, updated_file_content):
                        print("File content verification failed. Aborting.")
                        break

        # ProblemSolver methods
        brainstorming_response = problem_solver.brainstorm()
        conversation.append(("ProblemSolver", "Brainstorming Response: " + brainstorming_response))
        print(separator_line)
        print(f"\n{'-' * 50}\nProblemSolver - Brainstorming Response:\n{'-' * 50}\n{brainstorming_response}\n")
        print(separator_line)

        thought_response = problem_solver.thought()
        conversation.append(("ProblemSolver", "Thought Response: " + str(thought_response)))
        print(separator_line)
        print(f"\n{'-' * 50}\nProblemSolver - Thought Response:\n{'-' * 50}\n{thought_response}\n")
        print(separator_line)

        evaluation_response = problem_solver.evaluate()
        conversation.append(("ProblemSolver", "Evaluation Response: " + evaluation_response))
        print(separator_line)
        print(f"\n{'-' * 50}\nProblemSolver - Evaluation Response:\n{'-' * 50}\n{evaluation_response}\n")
        print(separator_line)

        thought_expand_response = problem_solver.thought_expand()
        conversation.append(("ProblemSolver", "Thought Expand Response: " + str(thought_expand_response)))
        print(separator_line)
        print(f"\n{'-' * 50}\nProblemSolver - Thought Expand Response:\n{'-' * 50}\n{thought_expand_response}\n")
        print(separator_line)

        expansion_response = problem_solver.expand()
        conversation.append(("ProblemSolver", "Expansion Response: " + expansion_response))
        print(separator_line)
        print(f"\n{'-' * 50}\nProblemSolver - Expansion Response:\n{'-' * 50}\n{expansion_response}\n")
        print(separator_line)

        decision_response = problem_solver.decide([brainstorming_response, evaluation_response, expansion_response])
        conversation.append(("ProblemSolver", "Decision Response: " + decision_response))
        print(separator_line)
        print(f"\n{'-' * 50}\nProblemSolver - Decision Response:\n{'-' * 50}\n{decision_response}\n")
        print(separator_line)

        # Coding agent loop after main_loops_before_coding full main loops
        if main_loop_count % main_loops_before_coding == 0:
            role_name, coding_agent, MessageClass, coding_inception_msg = agents[-1]
            # Gather previous agent messages excluding the current agent's own responses
            prev_agent_responses = [msg[1] for msg in conversation if msg[0] != role_name]

            # Find the previous main loop and refinement response by the coding agent
            prev_main_loop = None
            prev_refinement_response = None
            for agent_name, msg in reversed(conversation):
                if agent_name == coding_role_name:
                    if isinstance(msg, AIMessage):
                        prev_main_loop = msg.content
                elif prev_main_loop is not None and prev_refinement_response is None:
                    if isinstance(msg, AIMessage):
                        prev_refinement_response = msg.content
                    break

            # Gather most recent responses from user, assistant, thoughtful agents, and problem solver
            most_recent_responses = "\n".join([msg[1] for msg in conversation[-4:] if msg[0] != role_name])

            # Generate the updated file content based on the previous main loop, refinement, and most recent responses
            updated_file_content_prompt = (
                f"As the {coding_role_name}, based on the previous main loop, refinement, and most recent responses, please rewrite the file content with the improvement suggestions.\n\n"
                f"Most recent responses:\n{most_recent_responses}\n\n"
                f"{prev_main_loop}"
            )
            updated_file_content_ai_msg = coding_agent.step(MessageClass(content=updated_file_content_prompt))
            updated_file_content_msg = MessageClass(content=updated_file_content_ai_msg.content)
            conversation.append((role_name, updated_file_content_msg))
            total_tokens += len(updated_file_content_msg.content.split())
            print(separator_line)
            print(f"\n{'-' * 50}\n{role_name}:\n{'-' * 50}\n{updated_file_content_msg.content}\n")
            print(separator_line)

            # After you've received the response from the coding agent
            updated_file_content = updated_file_content_msg.content

            # Print the updated file content to the screen
            print(separator_line)
            print(f"\n{'-' * 50}\nUpdated File Content:\n{'-' * 50}\n{updated_file_content}\n")
            print(separator_line)

            # Write the updated file content to the file
            file_interactor.write_content(file_path, updated_file_content)

        # MonitorAgent intervention
        conversation_str = " ".join([str(msg[1]) for msg in conversation])
        if monitor_agent_class.should_intervene(conversation_str):
            intervention_message = monitor_agent_class.stage_intervention(conversation_str)
            print(f"\n{'-' * 50}\nMonitorAgent Intervention:\n{'-' * 50}\n{intervention_message}\n")
            conversation.append(("MonitorAgent", intervention_message))


write_conversation_to_file(conversation, 'conversation.txt')
