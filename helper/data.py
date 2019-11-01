#!/usr/bin/env python

import os
import json
import argparse
import tarfile
import io
from urllib.request import urlopen
import re
import warnings

from IPython.display import display, HTML, Audio, update_display
import ipywidgets as widgets
import appdirs

from basismixer.utils import pair_files

REPO_NAME = 'vienna4x22_rematched'
DATASET_BRANCH = 'master'
OWNER = 'OFAI'
DATASET_URL = 'https://api.github.com/repos/{}/{}/tarball/{}'.format(OWNER, REPO_NAME, DATASET_BRANCH)

# oggs will be downloaded from here
OGG_URL_BASE = 'https://spocs.duckdns.org/vienna_4x22/'

TMP_DIR = appdirs.user_cache_dir('basismixer')
# DATASET_DIR will be set to the path of our data
DATASET_DIR = None
PIECES = ()
PERFORMERS = ()
SCORE_PERFORMANCE_PAIRS = None

def get_datasetdir():
    """Get the SHA of the latest commit and return the corresponding
    datast directory path.
    
    """
    commit_url = ('https://api.github.com/repos/{}/{}/commits/{}'
                  .format(OWNER, REPO_NAME, DATASET_BRANCH))
    try:

        with urlopen(commit_url) as response:
            commit = json.load(response)

        repo_dirname = '{}-{}-{}'.format(OWNER, REPO_NAME, commit['sha'][:7])
        return os.path.join(TMP_DIR, repo_dirname)
        
    except Exception as e:
        warnings.warn('{} (url: {})'.format(e, commit_url))
        return None

    
def init_dataset():
    global DATASET_DIR, PIECES, PERFORMERS, SCORE_PERFORMANCE_PAIRS

    status = widgets.Output()
    display(status)
    status.clear_output()

    DATASET_DIR = get_datasetdir()
    
    if DATASET_DIR and os.path.exists(DATASET_DIR):
        status.append_stdout('Vienna 4x22 Corpus already downloaded.\n')
        status.append_stdout('Data is in {}'.format(DATASET_DIR))
    else:
        status.append_stdout('Downloading Vienna 4x22 Corpus...')
        try:
            with tarfile.open(fileobj=io.BytesIO(urlopen(DATASET_URL).read())) as archive:
                folder = next(iter(archive.getnames()), None)
                archive.extractall(TMP_DIR)
                # if folder:
                #     DATASET_DIR = os.path.join(TMP_DIR, folder)
                assert DATASET_DIR == os.path.join(TMP_DIR, folder)
                
        except Exception as e:
            status.append_stdout('\nError: {}'.format(e))
            return
        status.append_stdout('done\nData is in {}'.format(DATASET_DIR))
    
    folders = dict(musicxml=os.path.join(DATASET_DIR, 'musicxml'),
                   match=os.path.join(DATASET_DIR, 'match'))

    SCORE_PERFORMANCE_PAIRS = []
    paired_files = pair_files(folders)
    pieces = sorted(paired_files.keys())
    for piece in pieces:
        xml_fn = paired_files[piece]['musicxml'].pop()
        for match_fn in sorted(paired_files[piece]['match']):
            SCORE_PERFORMANCE_PAIRS.append((xml_fn, match_fn))
            
    fn_pat = re.compile('(.*)_(p[0-9][0-9])\.match')
    match_files = os.listdir(os.path.join(DATASET_DIR, 'match'))
    pieces, performers = zip(*[m.groups() for m in [fn_pat.match(fn)
                                                    for fn in match_files]
                               if m])
    PIECES = sorted(set(pieces))
    PERFORMERS = sorted(set(performers))

if __name__ == '__main__':
    init_dataset()
