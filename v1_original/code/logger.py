def save_log(text, path):
    with open(path, "a") as f:
        f.write(text + "\n")