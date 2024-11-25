import yaml
import json
import os

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        # Determine the file extension
        file_extension = os.path.splitext(self.config_path)[1]
        
        with open(self.config_path, 'r') as f:
            if file_extension in ['.yaml', '.yml']:
                config = yaml.load(f, Loader=yaml.FullLoader)
            elif file_extension in ['.json']:
                config = json.load(f)
            else:
                raise ValueError("Unsupported file type: {}".format(file_extension))
        return config

    def get_config(self):
        return self.config

    def get_config_path(self):
        return self.config_path

    def get_config_value(self, key):
        return self.config[key]

    def set_config_value(self, key, value):
        self.config[key] = value

    def save_config(self):
        file_extension = os.path.splitext(self.config_path)[1]
        with open(self.config_path, 'w') as f:
            if file_extension in ['.yaml', '.yml']:
                yaml.dump(self.config, f)
            elif file_extension in ['.json']:
                json.dump(self.config, f)
            else:
                raise ValueError("Unsupported file type: {}".format(file_extension))