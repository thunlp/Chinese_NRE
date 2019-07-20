import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import BiLstmEncoder
from .classifier import AttClassifier
from torch.autograd import Variable
from torch.nn import functional, init

class MGLattice_model(nn.Module):
    def __init__(self, data):
        super(MGLattice_model, self).__init__()
        # MG-Lattice encoder
        self.encoder = BiLstmEncoder(data)
        # Attentive classifier
        self.classifier = AttClassifier(data)

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, pos1_inputs, pos2_inputs, ins_label, scope):

        # ins_num * seq_len * hidden_dim
        hidden_out = self.encoder.get_seq_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, pos1_inputs, pos2_inputs)
        
        # batch_size * num_classes
        logit = self.classifier.get_logit(hidden_out, ins_label, scope)

        return logit


