import json
import os

import torch
import numpy as np
import subprocess
import soundfile
import tempfile

from IPython.display import display, Audio

from partitura import save_performance_midi, load_musicxml
from partitura.score import expand_grace_notes
from basismixer.predictive_models import FullPredictiveModel, construct_model
from basismixer.performance_codec import get_performance_codec
from basismixer.basisfunctions import make_basis


def render_midi(midi_fn):

    with tempfile.NamedTemporaryFile() as out_file:
        cmd = ['timidity', '-E', 'F', 'reverb=0', 'F', 'chorus=0',
               '--output-mono', '-Ov', '-o', out_file.name, midi_fn]
        try:
            ps = subprocess.run(cmd, stdout=subprocess.PIPE)
            if ps.returncode != 0:
                LOGGER.error('Command {} failed with code {}'
                             .format(cmd, ps.returncode))
                return False
        except FileNotFoundError as f:
            LOGGER.error('Executing "{}" returned  {}.'
                         .format(' '.join(cmd), f))
            return False
        data, fs = soundfile.read(out_file.name)
        aw = display(Audio(data=data, rate=fs, autoplay=True), display_id=True)
        return aw

    
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

def sanitize_performed_part(ppart):
    """Avoid negative durations in notes.

    """
    for n in ppart.notes:

        if n['note_off'] < n['note_on']:
            n['note_off'] == n['note_on']

        if n['sound_off'] < n['note_off']:
            n['sound_off'] == n['note_off']


def compute_basis_from_xml(xml_fn, input_names):
    # Load MusicXML file
    part = load_musicxml(xml_fn, force_note_ids=True)
    expand_grace_notes(part)

    # Compute basis functions
    _basis, bf_names = make_basis(part, list(set([bf.split('.')[0] for bf in input_names])))
    basis = np.zeros((len(_basis), len(input_names)))
    for i, n in enumerate(input_names):
        try:
            ix = bf_names.index(n)
        except ValueError:
            continue
        basis[:, i] = _basis[:, ix]

    return basis, part

