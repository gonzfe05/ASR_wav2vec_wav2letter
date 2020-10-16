from __future__ import absolute_import, division, print_function
# System imports
import os

def absoluteFilePaths(directory: str) -> list:
    """Generates absolute paths for the files in the given directory

    Args:
        directory (str): directory path

    Returns:
        list: Absolute paths to the files in the directory
    """
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return paths

def read_result(path: str) -> dict:
    """Read result of wav2letter decoder

    Args:
        path (str): Path to the hypothesis file

    Returns:
        dict: Trancriptions dict
    """
    trans = {}
    # read hypothesis file
    with open(path) as f:
        data = f.read().split('\n')
        data = data[:-1]
    
    for d in data:
        end = d.find('(')
        pred = d[0:end-1]
        file_name = d[end:].replace('(','').replace(')','')
        index = int(file_name.split('_')[-1].replace('.wav',''))
        base_name = int(file_name.split('_')[0])

        if base_name not in trans:
            trans[base_name] = [(index,pred)]
        else:
            trans[base_name].append((index,pred))

    for name in trans:
        trans[name] = sorted(trans[name], key = lambda x: x[0])
        trans[name] = ' '.join([t[1] for t in trans[name]])

    return trans