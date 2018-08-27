from contextlib import redirect_stdout

def save_model_summary(model, file_path):
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
