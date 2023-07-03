import os
import datetime
import re
from langchain.schema import AIMessage
import glob 
from transformers import GPT2Tokenizer
from config import conversation_directory

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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


# Check if the directory exists
if not os.path.exists(conversation_directory):
    # If not, create the directory
    os.makedirs(conversation_directory)

# Then you can use it as your workspace
os.chdir(conversation_directory)

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


# Function to create directories recursively if they don't already exist
def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")
    except Exception as e:
        print(f"Error while creating directory: {directory_path}. Error: {str(e)}")

## Function to recursively generate the file structure and scripts
def generate_file_structure_and_scripts(file_structure_content, coding_agent, project_directory="workspace", max_tokens=14000):
    os.makedirs(project_directory, exist_ok=True)

    lines = file_structure_content.split("\n")
    current_directory = project_directory
    indentation_levels = [0]

    for line in lines:
        stripped_line = line.lstrip()
        indentation = len(line) - len(stripped_line)

        if stripped_line.endswith(':'):
            # This is a directory
            directory_name = stripped_line[:-1]  # removing the colon at the end
            if directory_name.startswith("```"):  # ignore lines enclosed in triple backticks
                continue
            current_directory = os.path.join(current_directory, directory_name)
            os.makedirs(current_directory, exist_ok=True)
            indentation_levels.append(indentation)

        elif stripped_line and not stripped_line.startswith("```"):
            # This is a file
            while indentation < indentation_levels[-1]:  # Moving up in the directory tree
                current_directory = os.path.dirname(current_directory)
                indentation_levels.pop()

            file_name = stripped_line.strip('/').replace('/', '_')

            if file_name:  # This ignores empty lines
                file_path = os.path.join(current_directory, file_name)

                if not os.path.exists(file_path):
                    if stripped_line.endswith('/'):  # if the name ends with '/' treat it as a directory
                        os.makedirs(file_path, exist_ok=True)
                        print(f"Created directory: {file_path}")
                    else:
                        # Ensure parent directory exists
                        parent_directory = os.path.dirname(file_path)
                        os.makedirs(parent_directory, exist_ok=True)

                        code_prompt = f"As the {coding_agent}, provide code for the file with little to no placeholder code. This is meant to be a functional prototype and extensive: {file_name}"
                        code_prompt = truncate_text(code_prompt, max_tokens)
                        code_ai_msg = coding_agent.step(AIMessage(content=code_prompt))

                        if "```" in code_ai_msg.content:
                            code_content = "\n".join(code_ai_msg.content.split("```")[1].split("\n")[1:-1])  # Updated code extraction
                            # Remove placeholder end points
                            code_content = code_content.replace('...', '')
                        else:
                            print(f"Warning: AI response does not contain expected code block for file: {file_name}")
                            code_content = ""

                        with open(file_path, 'w') as f:
                            f.write(code_content)
                        print(f"Created file: {file_path}")

                if stripped_line.endswith('/'):  # Update current directory for the next file or directory
                    current_directory = file_path
                indentation_levels.append(indentation)

        # Check if we need to go up in directory tree
        if indentation < indentation_levels[-1]:
            while indentation < indentation_levels[-1]:
                current_directory = os.path.dirname(current_directory)
                indentation_levels.pop()

    # Now we prompt the Coding Agent to refine the created code
    for file_path in get_all_files_in_directory(project_directory):
        # Read the original code from the file
        with open(file_path, 'r') as file:
            original_code = file.read()

        # Ask the coding agent to refine the code
        refinement_prompt = f"As the {coding_agent}, please fill in all and any placeholder logic in the following code while expanding functionality when you can: \n\n{original_code}"
        refinement_prompt = truncate_text(refinement_prompt, max_tokens)
        refinement_ai_msg = coding_agent.step(AIMessage(content=refinement_prompt))

        # Extract the refined code from the AI response
        refined_code = refinement_ai_msg.content.split("```")[1].strip()  # Extract the code content only

        # Write the refined code back to the file
        with open(file_path, 'w') as file:
            file.write(refined_code)

        print(f"Refined file: {file_path}")


# Function to write code to a file
def write_code_to_file(file_path, code_content):
    # Check if the file path is a directory, if so, print message and return
    if os.path.isdir(file_path):
        print(f"Skipping directory: {file_path}")
        return

    # Check if the parent directory of the file path is a directory, if not, print message and return
    if not os.path.isdir(os.path.dirname(file_path)):
        print(f"Parent directory does not exist: {os.path.dirname(file_path)}")
        return

    # Check if the file already exists, if not, create and write to it
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            file.write(code_content)
        print(f"Generated code for file: {file_path}")
    else:
        print(f"Skipping existing file: {file_path}")


# Function to extract files from the file structure
def extract_files_from_file_structure(file_structure_content):
    files = []
    lines = file_structure_content.content.split("\n")
    current_directory = ""

    for line in lines:
        if line.startswith(" "):
            # File or subdirectory
            file_match = re.match(r"^\s+([├└──]+) (.+)", line)
            if file_match:
                indentation = file_match.group(1)
                file_name = file_match.group(2)
                path = os.path.join(current_directory, file_name) if current_directory else file_name
                files.append(path)
        else:
            # Directory
            directory_match = re.match(r"^([├└──]+) (.+)/$", line)
            if directory_match:
                indentation = directory_match.group(1)
                directory_name = directory_match.group(2)
                current_directory = os.path.join(current_directory, directory_name)

    return files

# Function to get all files in a directory, including nested directories
def get_all_files_in_directory(directory):
    return [f for f in glob.glob(directory + "**/*", recursive=True) if os.path.isfile(f)]



