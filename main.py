
from file_operations import write_conversation_to_file
from file_operations import generate_file_structure_and_scripts
from tot import ProblemSolver
from config import serpapi_api_key, conversation_directory 
from agents import user_role_name, assistant_role_name, thoughtful_role_name, coding_role_name
from agents import specified_task_msg, assistant_inception_prompt, thoughtful_inception_prompt, coding_inception_prompt
from langchain.prompts.chat import SystemMessage, HumanMessage, AIMessage
import openai
from agents import user_agent, user_inception_prompt, MonitorAgent, assistant_agent, coding_agent, thoughtful_agent, specified_task
from langchain.callbacks import get_openai_callback
from config import TOKEN_LIMIT, task

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
monitor_agent_class = MonitorAgent(task, "gpt-3.5-turbo-16k")


# Add agents to the list
agents = [
    (user_role_name, user_agent, HumanMessage, user_inception_msg),
    (assistant_role_name, assistant_agent, HumanMessage, assistant_inception_msg),
    (thoughtful_role_name, thoughtful_agent, AIMessage, thoughtful_inception_msg),
    (coding_role_name, coding_agent, AIMessage, coding_inception_msg),
]

# Set the number of loops for user, assistant, and thoughtful agents
loop_count = 3

# Set the number of main loops before running the coding agent and monitor agent intervention
main_loops_before_coding = 3
main_loops_before_monitor_intervention = 6

problem_solver = ProblemSolver(
    openai_key=openai.api_key, 
    serpapi_api_key=serpapi_api_key
)


# Main conversation loop
with get_openai_callback() as cb:
    chat_turn_limit = 50
    main_loop_count = 0

    # Set the goal for the ProblemSolver at the start of the conversation loop
    problem_solver.set_goal(specified_task_msg, conversation)

    for n in range(chat_turn_limit):
        separator_line = "\n" + "=" * 60 + "\n"

        # User, Assistant, Thoughtful loop
        for _ in range(loop_count):
            for i, (role_name, agent, MessageClass, inception_msg) in enumerate(agents[:-1]):
                if n == 1 and role_name == user_role_name:
                    ai_msg = agent.step(inception_msg)
                else:
                    # Gather previous agent messages excluding the current agent's own responses
                    prev_agent_responses = [msg[1] for msg in conversation if msg[0] != role_name]

                    # Filter out messages that are not strings or AIMessage objects
                    prev_agent_responses = [msg for msg in prev_agent_responses if isinstance(msg, str) or isinstance(msg, AIMessage)]

                    # Extract the content from each message and join them with a newline
                    message_content = "\n".join([msg.content if isinstance(msg, AIMessage) else msg for msg in prev_agent_responses[-2:]])
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

        final_product_response = problem_solver.produce_final_product(specified_task)
        conversation.append(("ProblemSolver", "Final Product Response: " + final_product_response))
        print(separator_line)
        print(f"\n{'-' * 50}\nProblemSolver - Final Product Response:\n{'-' * 50}\n{final_product_response}\n")
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
            
            # Generate the file structure and scripts based on the file structure content
            file_structure_prompt = (
                f"As the {coding_role_name}, based on the previous main loop, refinement, and most recent responses, please generate a hypothetical file structure "
                f"that would be suitable for this coding project. Refrain from any test files this is to be working prototype. There must be absolutely no test files or folders!\n\n"
                f"Most recent responses:\n{most_recent_responses}\n\n"
                f"{prev_main_loop}"
            )
            file_structure_ai_msg = coding_agent.step(MessageClass(content=file_structure_prompt))
            file_structure_msg = MessageClass(content=file_structure_ai_msg.content)
            conversation.append((role_name, file_structure_msg))
            total_tokens += len(file_structure_msg.content.split())
            print(separator_line)
            print(f"\n{'-' * 50}\n{role_name}:\n{'-' * 50}\n{file_structure_msg.content}\n")
            print(separator_line)

            # After you've received the response from the Python Coding Expert
            response = file_structure_msg.content  # Replace with actual response content

            # Extract the file structure content from the response
            file_structure_content = response.split('```')[1].strip() 

            # Generate file structure
            generate_file_structure_and_scripts(file_structure_content, coding_agent, conversation_directory)

            # Print message
            print(separator_line)
            print(f"\n{'-' * 50}\n{role_name}:\n{'-' * 50}\n{file_structure_msg.content}\n")
            print(separator_line)
            
        # MonitorAgent intervention
        conversation_str = " ".join([msg[1] for msg in conversation])
        if monitor_agent_class.should_intervene(conversation_str):
            intervention_message = monitor_agent_class.stage_intervention(conversation_str)
            print(f"\n{'-' * 50}\nMonitorAgent Intervention:\n{'-' * 50}\n{intervention_message}\n")
            conversation.append(("MonitorAgent", intervention_message))
                
    print(f"Total Successful Requests: {cb.successful_requests}")
    print(f"Total Tokens Used: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

write_conversation_to_file(conversation, 'conversation.txt')