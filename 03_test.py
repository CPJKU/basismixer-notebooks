#!/usr/bin/env python

import argparse
import numpy as np
import os
from basismixer import make_datasets
from basismixer.utils import load_pyc_bz, save_pyc_bz
from helper import init_dataset, data
from helper.predictions import construct_model, setup_output_directory, train_model, split_datasets

def main():
    parser = argparse.ArgumentParser(
        description="Train a Model given a dataset")
    parser.add_argument("--datasets", help=(
        'Path to pickled datasets file. If specified and the file exists, '
        'the `xmlfolder` and `matchfolder` options will be ignored, and it '
        'will be assumed that datasets in the specified file correspond to '
        'the model configuration. If specifed and the path does not exist, '
        'the datasets are computed and saved to the specified path.'))
    args = parser.parse_args()
    

    out_dir = setup_output_directory()
    
    model_config = [
        dict(onsetwise=False,
             basis_functions=['polynomial_pitch_basis',
                              'loudness_direction_basis',
                              'tempo_direction_basis',
                              'articulation_basis',
                              'duration_basis',
                              # my_basis,
                              'grace_basis',
                              'slur_basis',
                              'fermata_basis',
                              'metrical_basis'],
             parameter_names=['velocity_dev', 'timing', 'articulation_log'],
             # seq_len=1,
             # model=dict(constructor=['basismixer.predictive_models', 'FeedForwardModel'],
             #            args=dict(hidden_size=128)),
             seq_len=100,
             model=dict(constructor=['basismixer.predictive_models', 'RecurrentModel'],
                        args=dict(recurrent_size=128,
                                  n_layers=1,
                                  hidden_size=64)),
             train_args=dict(
                 optimizer=['Adam', dict(lr=1e-4)],
                 epochs=100,
                 save_freq=1,
                 early_stopping=100,
                 batch_size=1000,
             )
        ),
        # dict(onsetwise=True,
        #      basis_functions=['polynomial_pitch_basis',
        #                       'loudness_direction_basis',
        #                       'tempo_direction_basis',
        #                       'articulation_basis',
        #                       'duration_basis',
        #                       'slur_basis',
        #                       'fermata_basis',
        #                       'metrical_basis'],
        #      parameter_names=['velocity_trend', 'beat_period_standardized', 'beat_period_mean', 'beat_period_std'],
        #      seq_len=100,
        #      model=dict(constructor=['basismixer.predictive_models', 'RecurrentModel'],
        #                 args=dict(recurrent_size=128,
        #                           n_layers=1,
        #                           hidden_size=64)),
        #      train_args=dict(
        #          optimizer=['Adam', dict(lr=1e-4)],
        #          epochs=1,
        #          save_freq=1,
        #          early_stopping=100,
        #          batch_size=20,
        #      )
        # )
    ]
    init_dataset() # download the corpus if necessary; set some variables

    if data.DATASET_DIR is None:
        return
    
    # path to the MusicXML and Match files
    xmlfolder = os.path.join(data.DATASET_DIR, 'musicxml')
    matchfolder = os.path.join(data.DATASET_DIR, 'match')

    if args.datasets and os.path.exists(args.datasets):
        # LOGGER.info('Loading data from {}'.format(args.datasets))
        datasets = load_pyc_bz(args.datasets)
    else:
        datasets = make_datasets(model_config,
                                 xmlfolder,
                                 matchfolder)
        if args.datasets:
            # LOGGER.info('Saving data to {}'.format(args.datasets))
            save_pyc_bz(datasets, args.datasets)


    models = []
    test_sets = []
    for (dataset, in_names, out_names), config in zip(datasets, model_config):
        
        # Build model
        model, model_out_dir = construct_model(config, in_names, out_names, out_dir)
        # Split datasets
        train_set, valid_set, test_set = split_datasets(dataset)
        # Train Model
        train_model(model, train_set, valid_set, config, model_out_dir)
        
        models.append(model)
        test_sets.append(test_set)
    

if __name__ == '__main__':
    main()
