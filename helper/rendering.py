from partitura import save_performance_midi, load_musicxml
from partitura.score import expand_grace_notes
from basismixer.predictive_models import FullPredictiveModel, construct_model
from basismixer.performance_codec import get_performance_codec
import json
import os
import glob
import torch
import numpy as np

def load_model(models_dir):
    models = []
    for f in os.listdir(models_dir):
        path = os.path.join(models_dir, f)
        if os.path.isdir(path):
            model_config = json.load(open(os.path.join(path, 'config.json')))
            params = torch.load(os.path.join(path, 'best_model.pth'), 
                                map_location=torch.device('cpu'))['state_dict']
        
            model = construct_model(model_config, params)
            models.append(model)
    
    
    
    output_names = list(set([name for out_name in [m.output_names for m in models] for name in out_name]))
    input_names = list(set([name for in_name in [m.input_names for m in models] for name in in_name]))
    input_names.sort()
    output_names.sort()
    full_model = FullPredictiveModel(models, input_names, output_names)
    
    return full_model
