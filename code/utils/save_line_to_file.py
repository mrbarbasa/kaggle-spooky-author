from contextlib import redirect_stdout

def save_line_to_file(line, file_path, mode):
    with open(file_path, mode) as f:
        with redirect_stdout(f):
            print(line)
