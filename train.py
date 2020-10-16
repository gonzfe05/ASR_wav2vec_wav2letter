# Model imports
from wav2vec import EmbeddingDatasetWriter, Prediction
from utils.audio import preprocessing
from utils.text import normalize
# System imports
from utils.files import absolute_import, read_result, absoluteFilePaths
from os.path import abspath
import os
import argparse
import ntpath
import shutil
from shutil import copy2
# Audio manipulation imports
from pydub import AudioSegment
# Text manipulation imports
import string
import re
# Misc improts
import time
from multiprocessing import Pool
import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str,
                        required=True, help="Path to train file, eg. train.lst")
    parser.add_argument("--test_file", default=None, type=str,
                        required=True, help="Path to test file, eg. test.lst")
    parser.add_argument("--audio_path", default=None, type=str,
                        required=True, help="Path to wav files, eg. audio")
    parser.add_argument("--wav2vec_file", default=None, required=True,
                        type=str,help="Path to wav2vec model, check in resources/wav2vec.pt")
    parser.add_argument("--wav2letter", default=None, type=str,
                        required=True, help="Path to wav2letter library")
    parser.add_argument("--am_file", default=None, type=str,
                        help="Path to base aucostic model. If this is not given --> Training from scratch")
    parser.add_argument("--arch_file", default=None, type=str,
                        required=True, help="Path to archtecture file, check in resources/network.arch")
    parser.add_argument("--token_file", default=None, type=str,
                        required=True, help="Path to token file, check in resources/tokens.txt")
    parser.add_argument("--lexicon_file", default=None, type=str,
                        required=True, help="Path to lexicon file, check in resources/lexicon.txt")
    parser.add_argument("--output_path", default=None, type=str,
                        required=True, help="Output path for storing feature and model")
    parser.add_argument("--mode", default=None, type=str, required=True, 
                        help="Either 'finetune' or 'scratch'" 
                             "If scratch  --> train AM from scratch"
                             "If finetune --> continue training using AM provided by arch_file")
    parser.add_argument("--iter", default=50, type=int,
                        help="Number of interations. Set this number higher if training from scratch")
    parser.add_argument("--lr", default=0.5, type=float,
                        help="Learning rate, set 1.0 for training from scratch and 0.5 for fine-tunning")
    parser.add_argument("--lrcrit", default=0.004, type=float,
                        help="Learning rate crit, set 0.006 for training from scratch and 0.001 for fine-tunning ")
    parser.add_argument("--momentum", default=0.5, type=float,help="SGD momentum")
    parser.add_argument("--maxgradnorm", default=0.05, type=float,help="Max gradnorm")
    parser.add_argument("--nthread", default=1, type=int,help="Number of jobs")
    args = parser.parse_args()
    
    # Sanity checks
    if args.mode not in ['scratch','finetune']:
        raise ValueError('Training mode must be either scratch or finetune')
    if args.mode == 'finetune' and args.am_file == None:
        raise ValueError('For finetune training, am_file must be given')
    # Train AM from scratch
    if args.mode == 'scratch':
        args.lr = 1.0
        args.lrcrit = 0.01
        args.iter = 100
        args.momentum = 0.8
        args.maxgradnorm = 0.1

    start = time.time()
    # Parallell apply preprocessing on wav_files 
    audio_names = absoluteFilePaths(args.audio_path)
    preprocessed_path = os.path.join(args.audio_path, 'preprocessed')
    try:
        os.makedirs(preprocessed_path)
    except Exception as error:
        logger.error(f'Couldnt create preprocessed folder {preprocessed_path}: {error}')
        raise
    pool = Pool(4)
    pool.map(preprocessing, [(audio_names[i], preprocessed_path) for i in range(0,len(audio_names))])
    print("Preprocessing: ", time.time() - start)
    start = time.time()

    # Generates a nn.Module that extracts features in the forwardpass
    w2vec = Prediction(args.wav2vec_file)

    #Extract wav2vec features and write to disk
    feature_path = os.path.join(args.output_path,'feature')
    featureWritter = EmbeddingDatasetWriter(input_root = preprocessed_path,
                                            output_root = feature_path,
                                            loaded_model = w2vec, 
                                            extension="wav",use_feat=False)
    # featureWritter.write_features()

    # Read tsv file as a list of lists
    with open(args.train_file) as f:
        data = f.read().split('\n')
        data = [t for t in data if len(t) > 1]
        data = [d.split('\t') for d in data]

    # Build train.lst
    # It is a tsv with columns: `index, path_to_features, audio_length, trancription`
    for i in range(0,len(data)):
        path = os.path.join(preprocessed_path, data[i][0])
        text = normalize(data[i][1])
        audio = AudioSegment.from_wav(path)
        # Manipulate audio path to look for the corresponding features file
        path = os.path.join(feature_path,ntpath.basename(path))
        path = os.path.abspath(path)
        path = path.replace('.wav','.h5context')
        leng = str(len(audio) / 1000.0)
        idx = 'train' + str(i)
        data[i] = '\t'.join([idx, path, leng, text])
    # Write train.lst
    train_feature_file = abspath(os.path.join(args.output_path, 'train.lst'))
    with open(train_feature_file, 'w') as f:
        f.write('\n'.join(data))

    #Read test file as a list of lists
    with open(args.test_file) as f:
        data = f.read().split('\n')
        data = [t for t in data if len(t) > 1]
        data = [d.split('\t') for d in data]
    # Build test.lst
    # It is a tsv with columns: `index, path_to_features, audio_length, trancription`
    for i in range(0,len(data)):
        path = os.path.join(preprocessed_path, data[i][0])
        text = normalize(data[i][1])
        audio = AudioSegment.from_wav(path)
        # Manipulate audio path to look for the corresponding features file
        path = os.path.join(feature_path,ntpath.basename(path))
        path = os.path.abspath(path)
        path = path.replace('.wav','.h5context')
        leng = str(len(audio) / 1000.0)
        idx = 'test' + str(i)
        data[i] = '\t'.join([idx, path, leng, text])
    # Write test.lst
    test_feature_file = abspath(os.path.join(args.output_path, 'test.lst'))
    with open(test_feature_file, 'w') as f:
        f.write('\n'.join(data))
    
    # wav2letter tokens and arch paths 
    tokens = ntpath.basename(args.token_file)
    tokendirs = os.path.dirname(abspath(args.token_file))
    arch = ntpath.basename(args.arch_file)
    archdirs = os.path.dirname(abspath(args.arch_file))
    
    # Build wav2letter train command flags
    cmd = ['--runname=model']
    cmd.append('--rundir=' + abspath(args.output_path))
    cmd.append('--tokensdir=' + tokendirs)
    cmd.append('--tokens=' + tokens)
    cmd.append('--lexicon=' + args.lexicon_file)
    cmd.append('--archdir=' + archdirs)
    cmd.append('--arch=' + arch)
    cmd.append('--train=' + train_feature_file)
    cmd.append('--valid=' + test_feature_file)
    cmd.append('--lr=' + str(args.lr))
    cmd.append('--lrcrit=' + str(args.lrcrit))
    cmd.append('--iter=' + str(args.iter))
    cmd.append('--momentum=' + str(args.momentum))
    cmd.append('--maxgradnorm=' + str(args.maxgradnorm))
    cmd.append('--input=hdf5')
    cmd.append('--criterion=asg')    
    cmd.append('--linseg=1')
    cmd.append('--replabel=2')
    cmd.append('--onorm=target')
    cmd.append('--wnorm=true')
    cmd.append('--surround=|')
    cmd.append('--sqnorm=true')
    cmd.append('--mfsc=false')
    cmd.append('--wav2vec=true')
    cmd.append('--nthread=1')
    cmd.append('--batchsize=4')
    cmd.append('--transdiag=5')
    cmd.append('--melfloor=1.0')
    cmd.append('--minloglevel=0')
    cmd.append('--logtostderr=1')
    cmd.append('--enable_distributed=false')
    # Write wav2letter train command flags as a flagsfile
    cfg_path = os.path.join(args.output_path, 'fork_vec.cfg')
    with open(cfg_path,'w') as f:
        f.write('\n'.join(cmd))
    
    # Build wav2letter train command
    # For more info check https://github.com/facebookresearch/wav2letter/wiki/Train-a-model
    cmd = ['sudo']
    cmd.append(os.path.join(args.wav2letter, 'build/Train'))
    # Training supports three modes:
    # train : Train a model from scratch on the given training data.
    # continue : Continue training a saved model. This can be used for example to fine-tune with a smaller learning rate.
    #  The continue option makes a best effort to resume training from the most recent checkpoint of a given model as if there were no interruptions.
    # fork : Create and train a new model from a saved model. This can be used for example to adapt a saved model to a new dataset.
    if args.mode == 'finetune':
        # Correct usage of fork (see https://github.com/facebookresearch/wav2letter/issues/304#issuecomment-499201660):
        #   ./Train fork [path to old model.bin] [flags/params...]
        cmd.append('fork ' + abspath(args.am_file))
    else:
        cmd.append('train ')
    cmd.append('--flagsfile=' + cfg_path)
    cmd = ' '.join(cmd)
    # Run training
    print(f"We are going to run the following command:\n{cmd}")
    time.sleep(5)
    os.system(cmd)
    
    path_to_model = os.path.join(args.output_path, 'model/001_model_' + test_feature_file.replace('/','#') + '.bin')
    path_to_write = os.path.join(args.output_path, 'am.bin')
    assert os.path.exists(path_to_model), f"{path_to_model} not found, did the training failed?"
    
    copy2(path_to_model,path_to_write)
    
    # os.system('sudo rm -rf ' + os.path.join(args.output_path,'model'))
    # shutil.rmtree(os.path.join(args.output_path,'feature'), ignore_errors=True)
    # os.remove(test_feature_file)
    # os.remove(train_feature_file)
    # os.remove(cfg_path)

if __name__ == "__main__":
    main()
