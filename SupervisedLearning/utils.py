import numpy as np
import json
import os


def save_model(model, filepath):
    model_json = {}
    for i in range(len(model)):
        model_json[i] = model[i]
    
    with open(f'{filepath}.json', 'w') as file:
        json.dump(model_json, file)


def load_model(filepath):
    if os.path.exists(filepath) == False:
        return f'No such file at "{filepath}"'
    
    model = np.array(json.load(open(filepath, 'r')).values())
    return model
