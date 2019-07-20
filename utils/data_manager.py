import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import copy
import os
import configure
from .data import Data

def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []
    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)

def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data

def data_initialization(data, char_emb, gaz_file, train_file, dev_file, test_file, re2id_file, word_sense_map_file):

    data.build_words_larger_one_set(char_emb)

    # Build bidrectional maps for [words & word senses].
    data.build_word_sense_map(word_sense_map_file)

    # Build alphabet for train,dev,test and re2id files, including words, characters and bi-grams.
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_label_alphabet(re2id_file)

    # Build lexicon
    data.build_gaz_file(gaz_file)

    # Build matched words and senses alphabet.
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)

    data.fix_alphabet()
    return data

# Load data and print settings, the arguments are set in configure.py
def load_data(status='train'):
    public_path = configure.public_path

    # Path to load dataset (folder name)
    dataset = os.path.join(public_path,configure.dataset)

    # Path to load datasets (file name)
    train_file = os.path.join(dataset,configure.train_file)
    dev_file = os.path.join(dataset,configure.dev_file)
    test_file = os.path.join(dataset,configure.test_file)
    re2id_file = os.path.join(dataset,configure.relation2id)
    word_sense_map_file = os.path.join(public_path,configure.word_sense_map)

    # Set weights mode for each class in optimizer
    weights_mode = configure.weights_mode.lower() 
    gpu = torch.cuda.is_available()

    # Character Embeddings
    char_emb = os.path.join(public_path,configure.char_emb_file)
    # Bi-gram  Embeddings
    bichar_emb = None
    # Word sense Embeddings
    gaz_file = os.path.join(public_path,configure.sense_emb_file)

    # Print model settings
    print("CuDNN:", torch.backends.cudnn.enabled)
    print("GPU available:", gpu)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    print("Gaz file:",gaz_file)

    if status == 'train':
        data = Data()
        data.HP_use_char = False
        data.use_bigram = False
        data.norm_gaz_emb = False
        data.HP_fix_gaz_emb = False

        data.Encoder = configure.Encoder
        data.HP_gpu = gpu
        data.HP_batch_size = 1    
        data.gaz_dropout = 0.5    
        data.HP_lr = configure.lr
        data.set_maxlen(configure.max_length)

        data_initialization(data, char_emb, gaz_file, train_file, dev_file, test_file, re2id_file, word_sense_map_file)
        
        # Generate instances for train,dev and test files, the forms of instances are determined by load_mode
        data.generate_instance_with_gaz(train_file,'train',load_mode='ins')
        data.generate_instance_with_gaz(dev_file,'dev',load_mode='multilab-ins')
        data.generate_instance_with_gaz(test_file,'test',load_mode='multilab-ins')

        # Build weights for optimizer and pre-trained embeddings.
        data.build_weights(weights_mode)
        data.build_word_pretrain_emb(char_emb)
        data.build_biword_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)

    elif status == 'test':

        data = load_data_setting(configure.savedset)
        data.generate_instance_with_gaz(test_file,'test',load_mode='multilab-ins')

    return data
