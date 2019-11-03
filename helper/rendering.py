from partitura import save_performance_midi, load_musicxml
from partitura.score import expand_grace_notes
from basismixer.predictive_models import FullPredictiveModel, construct_model
from basismixer.performance_codec import get_performance_codec
import json
import os
import glob
import torch
import numpy as np
from basismixer.basisfunctions import make_basis

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

def compute_basis_from_xml(xml_fn, input_names):
    # Load MusicXML file
    part = load_musicxml(xml_fn, force_note_ids=True)
    expand_grace_notes(part)

    # Compute basis functions
    _basis, bf_names = make_basis(part, list(set([bf.split('.')[0] for bf in input_names])))
    basis_idx = np.array([int(np.where(input_names == bf)[0]) for bf in bf_names])
    basis = np.zeros((len(_basis), len(input_names)))
    basis[:, basis_idx] = _basis

    return basis, part
