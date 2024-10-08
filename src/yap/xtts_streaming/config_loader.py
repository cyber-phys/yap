import json

class ConfigLoader:
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
