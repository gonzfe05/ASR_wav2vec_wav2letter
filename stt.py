from __future__ import absolute_import, division, print_function
# Model imports
from wav2vec import EmbeddingDatasetWriter, Prediction
from utils.audio import preprocessing
# Audio manipulation imports
from pydub import AudioSegment
# System imports
from utils.files import read_result, absoluteFilePaths
import subprocess
import ntpath
import os
import sh
# Misc imports
import numpy as np
import time
from multiprocessing import Pool


class Transcriber:
    def __init__(self, w2letter, w2vec, am, tokens, lexicon, lm,
                 nthread_decoder = 1, lmweight = 1.51, wordscore = 2.57, beamsize = 200,
                 temp_path = './temp'):
        '''
        w2letter : path to complied wav2letter library (eg. /home/wav2letter)
        w2vec    : path to wav2vec model
        am       : path to aucostic model
        tokens   : path to graphmemes file
        lexicon  : path to dictionary file
        lm       : path to language model
        nthread_decoder: number of jobs for speeding up
        lmweight  : how much language model affect the result, the higher the more important
        wordscore : weight score for group of letter forming a word
        beamsize  : number of path for decoding, the higher the better but slower
        temp_path : directory for storing temporary files during processing
        '''
        
        self.w2letter = os.path.abspath(w2letter)
        self.am = os.path.abspath(am)
        self.tokens = ntpath.basename(tokens)
        self.tokensdir = os.path.dirname(os.path.abspath(tokens))
        self.lexicon = os.path.abspath(lexicon)
        self.lm = os.path.abspath(lm)
        self.nthread_decoder = nthread_decoder
        self.lmweight = lmweight
        self.wordscore = wordscore
        self.beamsize = beamsize
        self.pool = Pool(nthread_decoder)
        self.w2vec = Prediction(w2vec) #nn.Module for feature extraction
        self.output_path = os.path.abspath(temp_path)
        print(self.__dict__)
        
    def decode(self, input_file: str, output_path: str) -> str:
        """Generate and run command to Decode audio

        Args:
            input_file (str): test file path
            output_path (str): output directory

        Returns:
            str: Command as it was run
        """

        cmd = ['sudo']
        cmd.append(os.path.join(self.w2letter,'build/Decoder'))
        cmd.append('--am=' + self.am)
        cmd.append('--tokensdir=' + self.tokensdir)
        cmd.append('--tokens=' + self.tokens)
        cmd.append('--lexicon=' + self.lexicon)
        cmd.append('--lm=' + self.lm)
        cmd.append('--test=' + input_file)
        cmd.append('--sclite=' + str(output_path))
        cmd.append('--lmweight=' + str(self.lmweight))
        cmd.append('--wordscore=' + str(self.wordscore))
        cmd.append('--beamsize=' + str(self.beamsize))
        cmd.append('--beamthreshold=50')
        cmd.append('--silscore=0.0')
        cmd.append('--nthread_decoder=' + str(self.nthread_decoder))
        cmd.append('--smearing=max')
        cmd.append('--lm_memory=3000')
        cmd.append('---wav2vec=true')
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        process.wait()
        process.kill()
        return ' '.join(cmd)
        
    def transcribe(self,wav_files: list) -> list:
        """Generate transcriptions of audio files

        Args:
            wav_files (list): List of audio files paths

        Returns:
            list: List of transcriptions for those files
        """
        start = time.time()
        # Parallell apply preprocessing on wav_files 
        self.pool.map(preprocessing, [(wav_files[i], self.output_path) for i in range(0,len(wav_files))])
        print("Preprocessing: ", time.time() - start)
        start = time.time()
        
        # pre-compute and store embeddings for each audio
        # self.w2vec implements feature extraction as a forwardpass
        featureWritter = EmbeddingDatasetWriter(input_root = self.output_path,
                                                output_root = self.output_path,
                                                loaded_model = self.w2vec, 
                                                extension="wav",use_feat=False)
        featureWritter.write_features()

        print("Feature extraction: ", time.time() - start)
        start = time.time()
        
        # Get abosulte path to all the extracted embeddings
        paths = absoluteFilePaths(self.output_path)
        # Filter aout any non-embedding file
        paths = [p for p in paths if '.h5context' in p]
        # Create list of tsv lines
        lines = []
        for p in paths:
            file_name = ntpath.basename(p).replace('.h5context','')
            # The test list file (where id path duration transcription are stored, transcription can be empty)
            lines.append('\t'.join([file_name, p, '5', 'placeholder']))
        # Write tsv lines to test.lst
        with open(os.path.join(self.output_path, 'test.lst'),'w') as f:
            f.write('\n'.join(lines))

        #Decode embeddings on test.lst and save to output
        decode_res = self.decode(os.path.join(self.output_path, 'test.lst'),self.output_path)

        print("Decoding: ", time.time() - start)
        
        # Search for the generated hypothesis file in the output directory
        trans_file = None
        for path in absoluteFilePaths(self.output_path):
            if 'test.lst.hyp' in path:
                trans_file = path
        assert trans_file, f"An error occured during decoding.\nPlease run the following command in a separate terminal: {decode_res}"
        
        transcripts = read_result(trans_file)
        transcripts = list(transcripts.items())
        transcripts = sorted(transcripts, key = lambda x : x[0])
        transcripts = [t[1] for t in transcripts]
        
        # sh.rm(sh.glob(self.output_path + '/*'))
        
        return transcripts