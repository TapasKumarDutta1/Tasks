import re

def extract_number_from_string(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else 0

def load_ids(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f.readlines())
