import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .alphabet import Alphabet
from .functions import *
import pickle
from .gazetteer import Gazetteer
import re
import random


START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


# This class defines all settings about DATA
class Data:
    def __init__(self): 
        self.Encoder = 'MGLattice'
        self.MAX_SENTENCE_LENGTH = 200
        self.num_classes = 44
        self.pos_size = self.MAX_SENTENCE_LENGTH*2+3
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        
        # gaz is the external lexicon
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True
        self.char_features = "LSTM" 

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []

        self.train_freq = None
        self.dev_freq = None
        self.test_freq = None
        self.weights = None
        self.word_sense_map = None
        self.sense_word_map = None
        self.words_longer_than_one = None

        self.use_bigram = True
        self.word_emb_dim = 50
        self.biword_emb_dim = 50
        self.char_emb_dim = 30
        self.gaz_emb_dim = 50
        self.pos_emb_dim = 5
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0

        # hyper-parameters
        self.HP_iteration = 100
        self.HP_batch_size = 1
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = False
        self.HP_use_char = False
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Use          bigram: %s"%(self.use_bigram))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Biword alphabet size: %s"%(self.biword_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Gaz   alphabet size: %s"%(self.gaz_alphabet.size()))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Biword embedding size: %s"%(self.biword_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Gaz embedding size: %s"%(self.gaz_emb_dim))
        print("     Norm     word   emb: %s"%(self.norm_word_emb))
        print("     Norm     biword emb: %s"%(self.norm_biword_emb))
        print("     Norm     gaz    emb: %s"%(self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s"%(self.gaz_dropout))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Hyperpara  iteration: %s"%(self.HP_iteration))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("     Hyperpara     use_gaz: %s"%(self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s"%(self.HP_fix_gaz_emb))
        print("     Hyperpara    use_char: %s"%(self.HP_use_char))
        if self.HP_use_char:
            print("             Char_features: %s"%(self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def set_maxlen(self,maxlen):
        self.MAX_SENTENCE_LENGTH = maxlen
        self.pos_size = self.MAX_SENTENCE_LENGTH*2+3

    def build_label_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding='utf-8').readlines()
        for line in in_lines:
            if len(line)>2:
                try:
                    rel,id = line.strip().split(' ')
                except:
                    rel,id = line.strip().split('\t')
                self.label_alphabet.add(rel)
        self.label_alphabet_size = self.label_alphabet.size()
        self.num_classes = self.label_alphabet_size
        print('num_classes:',self.num_classes)

    # build weight of each relation label for loss function based on frequence
    def build_weights(self, mode='default'):
        weights = np.ones(self.num_classes)
        for label,cnt in self.train_freq.items():
            weights[self.label_alphabet.get_index(label)] += cnt
        if mode == 'reciprocal':
            self.weights = 1.0 / weights
        elif mode == 'smooth':
            self.weights = 1.0 / (weights**0.06)
        else:
            self.weights = np.ones(self.num_classes)

    # record word -> word sense & word sense -> word
    def build_word_sense_map(self, input=None):
        if input:
            fr = open(input,'r',encoding='utf-8')
            self.word_sense_map = dict()
            self.sense_word_map = dict()
            for line in fr:
                if len(line)<5:
                    continue
                line = line.strip().split(' ')
                self.word_sense_map[line[0]] = set(line[1:])
                for sense in line[1:]:
                    self.sense_word_map[sense] = line[0]

    # character whose length larger than one: <N> (refers to number)
    def build_words_larger_one_set(self, char_emb=''):
        self.words_longer_than_one = set()
        if char_emb:
            with open(char_emb,'r',encoding='utf-8') as file:
                
                for line in file:
                    line = line.strip()
                    if len(line) <= 1:
                        continue
                    tokens = line.split()
                    if len(tokens) <= 3:
                        continue
                    if len(tokens[0])>1:
                        self.words_longer_than_one.add(tokens[0])          

    # build alphabet for words, bi-grams and characters
    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding='utf-8').readlines()
        for idx in range(len(in_lines)):
            if len(in_lines[idx]) > 4:

                sent = in_lines[idx].strip().split('\t')[-1]
                sent = str2list(sent,self.words_longer_than_one)

                for widx,word in enumerate(sent):
                    if self.number_normalized:
                        word = normalize_word(word)
                    self.word_alphabet.add(word)

                    if widx < len(sent) - 1 and len(sent) > 2:
                        nxtword = sent[widx+1]
                        if self.number_normalized:
                            nxtword = normalize_word(nxtword)
                    else:
                        nxtword = NULLKEY

                    biword = word + nxtword

                    self.biword_alphabet.add(biword)
                    for char in word:
                        self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()

    # build gaz (lexcion)
    def build_gaz_file(self, gaz_file):
        if gaz_file:
            visit = set()
            fins = open(gaz_file, 'r',encoding='utf-8').readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                if fin:
                    if self.sense_word_map:
                        if fin in self.sense_word_map:
                            fin = self.sense_word_map[fin]
                        if fin in visit:
                            continue
                        visit.add(fin)
                    fin = str2list(fin,self.words_longer_than_one)
                    self.gaz.insert(fin, "one_source")
            print("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            print("Gaz file is None, load nothing")

    def build_gaz_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding='utf-8').readlines()
        word_list = []
        for line in in_lines:
            if len(line) > 4:

                sent = line.strip().split('\t')[-1]
                sent = str2list(sent,self.words_longer_than_one)
                for word in sent:
                    if self.number_normalized:
                        word = normalize_word(word)
                    word_list.append(word)
                w_length = len(word_list)
                for idx in range(w_length):
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                    for entity in matched_entity:
                        if self.gaz.space:
                            entity = ''.join(entity.split(self.gaz.space))
                        if self.word_sense_map and entity in self.word_sense_map:
                            for sense in self.word_sense_map[entity]:
                                self.gaz_alphabet.add(sense)
                        else:
                            self.gaz_alphabet.add(entity)
                word_list = []
        print("gaz alphabet size:", self.gaz_alphabet.size())


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close() 
        self.gaz_alphabet.close()  


    def build_word_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet, self.biword_emb_dim, self.norm_biword_emb)

    def build_gaz_pretrain_emb(self, emb_path):
        print("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet, self.gaz_emb_dim, self.norm_gaz_emb)

    def generate_instance_with_gaz(self, input_file, name, load_mode):
    
        if load_mode == 'ins':
            load_mode = MODE_INSTANCE
        elif load_mode == 'entpair':
            load_mode = MODE_ENTPAIR_BAG
        elif load_mode == 'relfact':
            load_mode = MODE_RELFACT_BAG
        elif load_mode == 'multilab-ins':
            load_mode = MODE_INSTANCE_MULTI_LABEL
            
        self.fix_alphabet()
        try:
            _ = self.words_longer_than_one
        except:
            self.words_longer_than_one = None
        if name == "train":
            self.train_texts, self.train_Ids, self.train_freq = read_instance_with_gaz_mode(name, input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.gaz_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH, self.num_classes, load_mode, self.word_sense_map, self.words_longer_than_one)
        elif name == "dev":
            self.dev_texts, self.dev_Ids, self.dev_freq = read_instance_with_gaz_mode(name, input_file, self.gaz,self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.gaz_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH, self.num_classes, load_mode, self.word_sense_map, self.words_longer_than_one)
        elif name == "test":
            self.test_texts, self.test_Ids, self.test_freq = read_instance_with_gaz_mode(name, input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.gaz_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH, self.num_classes, load_mode, self.word_sense_map, self.words_longer_than_one)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))

    def batch_iter(self, name, batch_size, shuffle=True, volatile_flag=False):
        if name == "train":
            instances = self.train_Ids
        elif name == "dev":
            instances = self.dev_Ids
        elif name == 'test':
            instances = self.test_Ids

        total_num = len(instances)
        total_batch = total_num//batch_size+1
        indices = [i for i in range(total_num)]
        if shuffle:
            random.shuffle(indices)
        for batch_id in range(total_batch):
            # for one batch
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size 
            if end >total_num:
                end =  total_num
            instance = []
            for ind in range(start,end):
                instance.append(instances[indices[ind]])
            if not instance:
                continue

            ins_num = 0 
            for bag in instance:
                ins_num += len(bag)

            word_seq_lengths = torch.LongTensor([0 for i in range(ins_num)])
            word_seq_tensor = autograd.Variable(torch.zeros((ins_num, self.MAX_SENTENCE_LENGTH)), volatile =  volatile_flag).long()
            biword_seq_tensor = autograd.Variable(torch.zeros((ins_num, self.MAX_SENTENCE_LENGTH)), volatile =  volatile_flag).long()
            ins_label_tensor = autograd.Variable(torch.zeros(ins_num),volatile =  volatile_flag).long()
            batch_label_tensor = autograd.Variable(torch.zeros(batch_size),volatile =  volatile_flag).long()
            pos1_seq_tensor = autograd.Variable(torch.zeros((ins_num, self.MAX_SENTENCE_LENGTH)), volatile =  volatile_flag).long()
            pos2_seq_tensor = autograd.Variable(torch.zeros((ins_num, self.MAX_SENTENCE_LENGTH)), volatile =  volatile_flag).long()
            mask = autograd.Variable(torch.zeros((ins_num, self.MAX_SENTENCE_LENGTH)),volatile =  volatile_flag).byte()
            scope = []
            chars = []
            batch_labels = []
            gazs = []
            
            idx = 0
            for bid,bag in enumerate(instance):
                scope.append([idx,idx+len(bag)])
                batch_label_tensor[bid] = bag[0][4][0]
                bag_labels = []
                for ins in bag:
                    seq, biseq, chr, gaz, label, pos1, pos2 = ins
                    seqlen = len(seq)
                    word_seq_lengths[idx] = seqlen
                    chars.append(chr)
                    gazs.append(gaz)

                    bag_labels += label
                    label = label[0]
                    ins_label_tensor[idx] = label

                    word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
                    biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)

                    pos1 = pos1 + [2*self.MAX_SENTENCE_LENGTH+2 for i in range(self.MAX_SENTENCE_LENGTH-seqlen)]
                    pos2 = pos2 + [2*self.MAX_SENTENCE_LENGTH+2 for i in range(self.MAX_SENTENCE_LENGTH-seqlen)]
                    pos1_seq_tensor[idx, : ] = torch.LongTensor(pos1)
                    pos2_seq_tensor[idx, : ] = torch.LongTensor(pos2)

                    mask[idx, :seqlen] = torch.Tensor([1 for i in range(seqlen)])
                    
                    idx += 1
                batch_labels.append(bag_labels)
            
            ### deal with char
            pad_chars = [chars[idx] + [[0]] * (self.MAX_SENTENCE_LENGTH-len(chars[idx])) for idx in range(len(chars))]
            length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
            max_word_len = max(list(map(max, length_list)))
            char_seq_tensor = autograd.Variable(torch.zeros((ins_num, self.MAX_SENTENCE_LENGTH, max_word_len)), volatile =  volatile_flag).long()
            char_seq_lengths = torch.LongTensor(length_list)
            for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
                for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                    # print len(word), wordlen
                    char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
            char_seq_tensor = char_seq_tensor.view(ins_num*self.MAX_SENTENCE_LENGTH,-1)
            char_seq_lengths = char_seq_lengths.view(ins_num*self.MAX_SENTENCE_LENGTH,)
            char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
            char_seq_tensor = char_seq_tensor[char_perm_idx]
            _, char_seq_recover = char_perm_idx.sort(0, descending=False)
            
            gaz_list = []
            for i in range(len(gazs)):
                gazlen = len(gazs[i])
                gaz_list.append(gazs[i] + [[] for j in range(self.MAX_SENTENCE_LENGTH-gazlen)])
            gaz_list = [ gaz_list[0], volatile_flag ]
            if self.HP_gpu:
                word_seq_tensor = word_seq_tensor.cuda()
                biword_seq_tensor = biword_seq_tensor.cuda()
                word_seq_lengths = word_seq_lengths.cuda()
                ins_label_tensor = ins_label_tensor.cuda()
                batch_label_tensor = batch_label_tensor.cuda()
                pos1_seq_tensor = pos1_seq_tensor.cuda()
                pos2_seq_tensor = pos2_seq_tensor.cuda()
                char_seq_tensor = char_seq_tensor.cuda()
                char_seq_recover = char_seq_recover.cuda()
                mask = mask.cuda()

            if name != 'train':
                batch_label_tensor = batch_labels # for evaluation (answer is multi-label)

            yield (gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, char_seq_tensor, char_seq_lengths, char_seq_recover, pos1_seq_tensor, pos2_seq_tensor, ins_label_tensor, batch_label_tensor, mask, scope)
