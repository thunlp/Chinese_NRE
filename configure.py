savemodel = "models/SanWen.pkl"
loadmodel = "models/SanWen.pkl-233"
savedset = "models/SanWen.pkl.dset"

public_path = "data"
dataset = "SanWen"
train_file = "train.txt"
dev_file = "valid.txt"
test_file = "test.txt"
relation2id = "relation2id.txt"
char_emb_file = "vec.txt"
sense_emb_file = "sense.txt"
word_sense_map = "sense_map.txt"
max_length = 86

Encoder = 'MGLattice' # 'MGLattice' or 'GRU'
Optimizer = 'SGD' # 'SGD' or 'Adam'
lr = 0.015 # recommend: 0.015 for SGD ( with lr decay ) and 0.0005 for Adam
weights_mode = 'smooth'