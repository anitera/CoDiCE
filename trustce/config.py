import yaml

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
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
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)