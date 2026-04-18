"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import tqdm
import scipy.io.wavfile as wavfile
import torch as th
import numpy as np


class BinauralDataset:

    '''
    dataset_directory: (str) base directory of the dataset
    chunk_size_ms: (int) length of an acoustic chunk in ms
    overlap: (float) overlap ratio between two neighboring audio chunks, must be in range [0, 1)
    '''
    def __init__(self,
                 dataset_directory,
                 chunk_size_ms=200,
                 overlap=0.5,
                 exclude_subjects=None
                 ):
        super().__init__()
        # load audio data and relative transmitter/receiver position/orientation
        self.mono, self.binaural, self.view = [], [], []
        
        # dynamic subject detection
        import glob
        import os
        subject_dirs = sorted(glob.glob(f"{dataset_directory}/subject*"))
        if not subject_dirs:
             raise ValueError(f"No subject directories found in {dataset_directory}")
             
        pbar = tqdm.tqdm(subject_dirs)
        
        if exclude_subjects is None:
            exclude_subjects = []
            
        for subject_path in pbar:
            subject_name = os.path.basename(subject_path)
            
            # Check for exclusion
            # supports both full name "subject4" or just integer 4
            if subject_name in exclude_subjects:
                pbar.set_description(f"Skipping {subject_name}")
                continue
            if any(str(sub) in subject_name for sub in exclude_subjects if isinstance(sub, int)):
                 pbar.set_description(f"Skipping {subject_name}")
                 continue
                 
            pbar.set_description(f"loading data: {subject_name}")
            
            sr, mono_np = wavfile.read(f"{subject_path}/mono.wav")
            sr, binaural_np = wavfile.read(f"{subject_path}/binaural.wav")
            
            # Convert to torch tensor and ensure shape (Channels, Time)
            # scipy reads as (Time, Channels) or just (Time) if mono
            if mono_np.ndim == 1:
                mono_np = mono_np[None, :] # Add channel dim: (1, T)
            else:
                mono_np = mono_np.T # (T, C) -> (C, T)
                
            if binaural_np.ndim == 1:
                binaural_np = binaural_np[None, :]
            else:
                binaural_np = binaural_np.T

            # Normalize if integer type
            if np.issubdtype(mono_np.dtype, np.integer):
                if mono_np.dtype == np.int16:
                    mono_np = mono_np.astype(np.float32) / 32768.0
                elif mono_np.dtype == np.int32:
                    mono_np = mono_np.astype(np.float32) / 2147483648.0
                elif mono_np.dtype == np.uint8:
                    mono_np = (mono_np.astype(np.float32) - 128) / 128.0
                else:
                    # Fallback for other integer types
                    max_val = np.iinfo(mono_np.dtype).max
                    mono_np = mono_np.astype(np.float32) / max_val
            else:
                mono_np = mono_np.astype(np.float32)

            if np.issubdtype(binaural_np.dtype, np.integer):
                if binaural_np.dtype == np.int16:
                    binaural_np = binaural_np.astype(np.float32) / 32768.0
                elif binaural_np.dtype == np.int32:
                    binaural_np = binaural_np.astype(np.float32) / 2147483648.0
                elif binaural_np.dtype == np.uint8:
                    binaural_np = (binaural_np.astype(np.float32) - 128) / 128.0
                else:
                    max_val = np.iinfo(binaural_np.dtype).max
                    binaural_np = binaural_np.astype(np.float32) / max_val
            else:
                binaural_np = binaural_np.astype(np.float32)

            mono = th.from_numpy(mono_np)
            binaural = th.from_numpy(binaural_np)

            # receiver is fixed at origin in this dataset, so we only need transmitter view
            tx_view = np.loadtxt(f"{subject_path}/tx_positions.txt").transpose()
            self.mono.append(mono)
            self.binaural.append(binaural)
            self.view.append(tx_view.astype(np.float32))
        # ensure that chunk_size is a multiple of 400 to match audio (48kHz) and receiver/transmitter positions (120Hz)
        self.chunk_size = chunk_size_ms * 48
        if self.chunk_size % 400 > 0:
            self.chunk_size = self.chunk_size + 400 - self.chunk_size % 400
        # compute chunks
        self.chunks = []
        for subject_id in range(len(self.mono)):
            last_chunk_start_frame = self.mono[subject_id].shape[-1] - self.chunk_size + 1
            hop_length = int((1 - overlap) * self.chunk_size)
            for offset in range(0, last_chunk_start_frame, hop_length):
                self.chunks.append({'subject': subject_id, 'offset': offset})

    def __len__(self):
        '''
        :return: number of training chunks in dataset
        '''
        return len(self.chunks)

    def __getitem__(self, idx):
        '''
        :param idx: index of the chunk to be returned
        :return: mono audio as 1 x T tensor
                 binaural audio as 2 x T tensor
                 relative rx/tx position as 7 x K tensor, where K = T / 400 (120Hz tracking vs. 48000Hz audio)
        '''
        subject = self.chunks[idx]['subject']
        offset = self.chunks[idx]['offset']
        mono = self.mono[subject][:, offset:offset+self.chunk_size]
        binaural = self.binaural[subject][:, offset:offset+self.chunk_size]
        view = self.view[subject][:, offset//400:(offset+self.chunk_size)//400]
        return mono, binaural, view
