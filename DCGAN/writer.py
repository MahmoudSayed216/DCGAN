class Writer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.file_path = base_path+"/logs.txt"
    def save_configs(self, cfg):
        pass
    def add_line(self, line):
        file = open(self.file_path, "+a")
        file.write(line+"\n")
        file.close()
    
    def add_scaler(self, tag: str, step: int, value):
        file = open(self.file_path, "+a")
        line = f"step {step}: {tag} = {value}\n"
        file.write(line)
        file.close()

    