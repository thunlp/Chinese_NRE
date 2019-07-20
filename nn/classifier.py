import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import functional, init

# Implementation of Classifier
class AttClassifier(nn.Module):
    def __init__(self, data):
        super(AttClassifier, self).__init__()
        self.gpu = data.HP_gpu
        self.num_steps = data.MAX_SENTENCE_LENGTH
        # Number of classes of labels, which is Y in the paper
        self.num_classes = data.num_classes
        # Size of hidden state, which is d^h
        self.gru_size = data.HP_hidden_dim
        self.batch_size = data.HP_batch_size
        self.att_drop = nn.Dropout(data.HP_dropout)

        # Parameters for attention classifier
        # Weighted matrix W
        self.attention_w = nn.Parameter(torch.FloatTensor(self.gru_size, 1), requires_grad = True) # w
        # Embeddings of labels
        self.relation_embedding = nn.Parameter(torch.FloatTensor(self.num_classes,self.gru_size), requires_grad = True) # W
        # Bias term b
        self.sen_d = nn.Parameter(torch.FloatTensor(self.num_classes), requires_grad = True) # b

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.attention_w)
        init.constant_(self.sen_d, val = 0)
        init.normal_(self.relation_embedding)
        
    def __attention_train_logit__( self, x, query ): # query: ins_label
        current_relation = self.relation_embedding[query]
        attention_logit = torch.sum(current_relation*x,-1)
        return attention_logit
    
    def __attention_test_logit__( self, x ):
        attention_logit = torch.matmul(x,torch.transpose(self.relation_embedding,0,1))
        return attention_logit
        
    def __logit__( self, x ):
        # [batch_size * gru_size] /dot [gru_size * num_classes] -> [batch_size * num_classes]
        logit = torch.matmul(x,torch.transpose(self.relation_embedding,0,1))+self.sen_d
        logit = logit.view([x.size(0),self.num_classes]) # [batch_size * num_classes]
        return logit

    def get_logit(self, hidden_out, ins_label, scope):
        
        ins_num = hidden_out.size(0)
        # Hidden state, H in the paper.
        H = torch.tanh(hidden_out)
        H = H.view([ins_num, self.num_steps, self.gru_size])
        # Weights of h
        alpha = torch.matmul(H, self.attention_w)
        alpha = alpha.view([ins_num, self.num_steps])
        alpha = torch.nn.functional.softmax(alpha, dim = 1)
        alpha = alpha.view([ins_num, 1, self.num_steps])
        # h* in the paper
        h_star = torch.matmul(alpha, hidden_out)
        h_star = h_star.view([ins_num, self.gru_size])

        attention_r = h_star   
         
        # Selector
        if self.training:
            attention_r = self.att_drop(attention_r)
            attention_logit = self.__attention_train_logit__(attention_r,ins_label) # [ins_num * gru_size]
            bag_repre = []
            for i in range(self.batch_size):
                bag_hidden_mat = torch.tanh(attention_r[scope[i][0]:scope[i][1]]) # [bag_size * gru_size]
                bag_size = scope[i][1]-scope[i][0]
                # Attention scores 
                attention_score = torch.nn.functional.softmax(attention_logit[scope[i][0]:scope[i][1]],-1).view(1,bag_size) # [bag_size * gru_size] -> [bag_size]
                # Bag representations
                bag_repre.append(torch.matmul(attention_score,bag_hidden_mat).view(self.gru_size)) # [1 * gru_size]
            bag_repre = torch.stack(bag_repre) 
            # [batch_size * gru_size] /dot [gru_size * num_classes] -> [batch_size * num_classes]
            logit = self.__logit__(bag_repre)
        else:
            attention_logit = self.__attention_test_logit__(attention_r) # [ins_num * num_classes]
            bag_logit = []
            for i in range(self.batch_size):                
                bag_hidden_mat = torch.tanh(attention_r[scope[i][0]:scope[i][1]]) # [bag_size * gru_size]
                bag_size = scope[i][1]-scope[i][0]
                # Scores of each sentence for each class
                attention_score = torch.nn.functional.softmax(torch.transpose(attention_logit[scope[i][0]:scope[i][1],:],0,1),-1).view(self.num_classes,bag_size)
                # Merge multiple sentence representations
                bag_repre_for_each_rel = torch.matmul(attention_score,bag_hidden_mat)
                bag_logit_for_each_rel = self.__logit__(bag_repre_for_each_rel) # [num_classes * num_classes]
                bag_logit.append(torch.diag(torch.nn.functional.softmax(bag_logit_for_each_rel,-1)))
            logit = torch.stack(bag_logit)
            
        logit = logit.view([self.batch_size,self.num_classes])

        # Scores for each class (without normalization)
        return logit 



