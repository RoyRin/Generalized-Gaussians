import yaml

def open_yaml(filename):
    with open(filename, 'r') as stream:
        return yaml.safe_load(stream)

   
def save_yaml(d, filepath):
    """write a dictionary to a yaml file

    Args:
        d (_type_): _description_
        filepath (_type_): _description_
    """
    with open(filepath, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)

def write_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)
