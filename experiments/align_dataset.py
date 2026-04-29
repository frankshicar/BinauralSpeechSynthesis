"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import glob
import argparse
import shutil
import numpy as np
import scipy.io.wavfile as wavfile
import tqdm

from src.alignment import find_alignment_offset, align_signals, diagnose_alignment

def align_subject(subject_path, output_path, dry_run=False):
    """
    Align audio for a single subject and save to output_path.
    """
    subject_name = os.path.basename(subject_path)
    
    # ensure output directory exists
    if not dry_run:
        os.makedirs(output_path, exist_ok=True)
    
    # Load audio
    mono_file = os.path.join(subject_path, "mono.wav")
    binaural_file = os.path.join(subject_path, "binaural.wav")
    
    if not os.path.exists(mono_file) or not os.path.exists(binaural_file):
        print(f"Skipping {subject_name}: Missing audio files.")
        return

    sr_mono, mono = wavfile.read(mono_file)
    sr_bin, binaural = wavfile.read(binaural_file)
    
    # Check Sample Rate
    if sr_mono != sr_bin:
        raise ValueError(f"Sample rate mismatch in {subject_name}: mono={sr_mono}, binaural={sr_bin}")
    
    sample_rate = sr_mono
    
    # Handle Data Types (Normalize to float32 for processing)
    # alignment functions expect float or compatible numpy arrays
    # We will just pass the raw numpy arrays, src.alignment handles basic flattening/slicing
    # But for writing back, we want to maintain quality or consistent format (float32)
    
    # Transpose to (Chunks, Time) which is what src/alignment expects for multi-channel
    if mono.ndim == 1:
        mono = mono[None, :] # (1, T)
    else:
        mono = mono.T # (C, T)
        
    if binaural.ndim == 1:
        binaural = binaural[None, :]
    else:
        binaural = binaural.T # (2, T)
        
    # Calculate Offset
    # Increase search range to 200ms (9600 samples) to avoid boundary effects
    offset, corr = find_alignment_offset(mono, binaural, max_shift=9600, sample_rate=sample_rate)
    
    # Align
    aligned_mono, aligned_binaural = align_signals(mono, binaural, offset)
    
    # Enforce min length again just to be safe (User Suggestion 1)
    min_len = min(aligned_mono.shape[1], aligned_binaural.shape[1])
    aligned_mono = aligned_mono[:, :min_len]
    aligned_binaural = aligned_binaural[:, :min_len]

    # Convert back to (Time, Channels) for saving
    aligned_mono_save = aligned_mono.T 
    aligned_binaural_save = aligned_binaural.T
    
    # Squeeze mono if it was 1D originally? 
    # Scipy write handles (T, 1) fine.
    
    diagnose_alignment(mono, binaural, offset, corr, sample_rate)
    
    if not dry_run:
        # Save WAV
        wavfile.write(os.path.join(output_path, "mono.wav"), sample_rate, aligned_mono_save)
        wavfile.write(os.path.join(output_path, "binaural.wav"), sample_rate, aligned_binaural_save)
        
        # Copy Position File (User Suggestion 2: "Unmodified")
        src_pos = os.path.join(subject_path, "tx_positions.txt")
        dst_pos = os.path.join(output_path, "tx_positions.txt")
        if os.path.exists(src_pos):
            shutil.copy2(src_pos, dst_pos)
        else:
            print(f"Warning: {src_pos} not found.")

def main():
    parser = argparse.ArgumentParser(description="Align Dataset Audio")
    parser.add_argument("--dataset_directory", type=str, default="./dataset/trainset")
    parser.add_argument("--output_directory", type=str, default="./dataset/trainset_aligned")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    
    # Glob subjects
    subject_dirs = sorted(glob.glob(os.path.join(args.dataset_directory, "subject*")))
    if not subject_dirs:
        print(f"No subjects found in {args.dataset_directory}")
        return
        
    print(f"Found {len(subject_dirs)} subjects.")
    print(f"Output Directory: {args.output_directory}")
    if args.dry_run:
        print("DRY RUN MODE: No files will be written.")
    
    for subject_dir in tqdm.tqdm(subject_dirs):
        subject_name = os.path.basename(subject_dir)
        output_dir_subj = os.path.join(args.output_directory, subject_name)
        
        print(f"\nProcessing {subject_name}...")
        try:
            align_subject(subject_dir, output_dir_subj, args.dry_run)
        except Exception as e:
            print(f"Error processing {subject_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
