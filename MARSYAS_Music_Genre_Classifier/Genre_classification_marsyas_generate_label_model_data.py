# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:30:46 2021

@author: prasad
"""
import json 
import os   
import librosa
import math

DEBUG_PRINT = False
#sample rate as per dataset info
SAMPLE_RATE = 22050
FILE_DURATION=30
NUM_SAMPLES= SAMPLE_RATE*FILE_DURATION

#user defined 
NUM_FILES_SELECTED_PER_GENRE=10
NUM_SEGMENTS = 5
FFT_LEN =2048
NUM_MFCC_VECTORS_PER_BLOCK = 13
HOP_LENGTH= int(FFT_LEN/4)
BLOCK_SIZE=int(NUM_SAMPLES/NUM_SEGMENTS)


#dataset is in same directory as script
dataset_directory = "genres"
filepath=""

dir_list =[]
json_path = "marsyas_directory.json"
json_dataset="marsyas_mfcc_dat.json"
#dictionary of parsed marsyas data to store as a json
data={
      "genres":[],
      "mfcc": [],
      "labels":[]
      }

directory={
    "directories": [],
    "files":{},
    }


def write_marsyas_directory_to_json(root,json_path):
    for i , (dirpath, dirname, filename) in enumerate(os.walk(dataset_directory)):
        if(dirpath!=root):
            if(DEBUG_PRINT):
                print("dir path is {}\n".format(dirpath))
            files=[]
            for file in filename:
                filepath = dirpath+ "\\" + file
                if(DEBUG_PRINT):
                    print("\nfilepaths are:"+ filepath)
                files.append(filepath)
            directory["files"][directory["directories"][i-1]]=files
            
        else:
            for dirs in dirname:
                directory["directories"].append(dirs)
                directory["files"][dirs]=[]
    
    json_directory = json.dumps(directory,indent=4)
    with open(json_path, "w") as outfile:
        outfile.write(json_directory)


def extract_mfcc_vectors(json_path,num_files_per_genre, directory,mfcc_size,segments,block_size,sample_rate,hop_length,fft_len):
    
    expected_mfcc_size = math.ceil(block_size/hop_length)
    
    for i, genre in enumerate(directory["directories"]): 
        data["genres"].append(genre)
        for j,file in enumerate(directory["files"][genre]):
            if(j<num_files_per_genre):
                audio_file,sr = librosa.load(file,sr=SAMPLE_RATE)
                for segment in range(segments):
                    start_offset= block_size*segment
                    end_offset=start_offset + block_size
                    mfcc_vect= librosa.feature.mfcc(audio_file[start_offset:end_offset],sr,n_mfcc=mfcc_size,n_fft=fft_len,hop_length=hop_length)
                    mfcc_vect=mfcc_vect.T
                    if(len(mfcc_vect)==expected_mfcc_size):
                        data["mfcc"].append(mfcc_vect.tolist())
                        data["labels"].append(i)    
                
    json_directory = json.dumps(data,indent=4)
    with open(json_path, "w") as outfile:
        outfile.write(json_directory)

write_marsyas_directory_to_json(dataset_directory,json_path)
extract_mfcc_vectors(json_dataset,NUM_FILES_SELECTED_PER_GENRE,directory,NUM_MFCC_VECTORS_PER_BLOCK, NUM_SEGMENTS, BLOCK_SIZE,SAMPLE_RATE,HOP_LENGTH,FFT_LEN)