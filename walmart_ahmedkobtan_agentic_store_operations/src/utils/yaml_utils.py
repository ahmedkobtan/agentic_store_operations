import yaml

def read_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    return y
