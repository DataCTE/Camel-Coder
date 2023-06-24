import os

class FileInteractor:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_files(self):
        file_list = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                if os.path.isfile(full_path):  # Check if it is a file, regardless of the extension
                    file_list.append(full_path)
        return file_list

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def get_lines(self, file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def save_file(self, file_path, content):
        with open(file_path, 'w') as file:
            file.write(content)

    def write_content(self, file_path, content):
        with open(file_path, 'a') as file:
            file.write(content)

    def verify_file_content(self, file_path, expected_content):
        current_content = self.read_file(file_path)
        return current_content.strip() == expected_content.strip()