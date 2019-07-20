# Chinese-NRE

Source code for ACL 2019 paper "Chinese Relation Extraction with Multi-Grained Information and External Linguistic Knowledge".  Some code in this repository is based on the excellent open-source project https://github.com/jiesutd/LatticeLSTM. The paper will be published by the official process of ACL2019.



## Requirements

- Python 3.6
- Pytorch 0.4.1



## Datasets

Three datasets are used in our paper:

- **FinRE**:  A manual-labeled financial news RE dataset. The data cannot be made public for the time being.


- **SanWen**:  A  Chinese literature NER-RE dataset, the source of the dataset is  [https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset). 
- **ACE 2005**:  A benchmark RE dataset. According to the terms of **LDC**, we are not allowed to share the dataset with the third party. If you have the LDC license, please obtain the dataset (LDC2006T06) and follow the data format by yourself.

In this project, `train.txt` , `dev.txt` and `test.txt` are all from **SanWen**.



## Data Format

### Input Format

**`data/SanWen/train.txt, dev.txt, test.txt`** One instance per line with 4 columns separated by tab character. The first and second columns are head and tail entities. The third column is the relation label and the last one is text:

```
[head]	[tail]	[relation]	  text
```

For example ( one line ):

```
 湖底	   卵石	 Located	 连湖底的卵石颜色也可分辨
```

**`data/SanWen/relation2id.txt`** One relation per line with 2 columns separated by tab character. The first column is teh label while the second one is the corresponding ID:

```
[relation]	[ID]
```

### Pre-trained Character Embeddings

**`data/vec.txt`** One character per line. For each line, the first column is the character, the rest columns is the value of the embedding of the character.

### Pre-trained Word-Sense Embeddings

**`data/sense.txt`** Similar to character embedding but for word senses. For example:

```
释放#1 0.304095 ...
释放#2 -0.175496 ...
夏天 -0.230772 ...
```

Here, **A#n** means that it is the **n**-th sense of word **A** ( **A** is a polysemous word ).  And the word-sense embeddings could be trained by the [SAT](https://github.com/thunlp/SE-WRL-SAT) (Sememe Attention over Target) approach.

### Word-Sense Map

**`data/sense_map.txt`** Recording all senses for each polysemous word, corresponding to the word sense embedding.  One word per line, for each line, the first column is the word, and the rest columns are all the senses of it ( if exist ). For example:

```
释放 释放#1 释放#2
夏天
```

The **`sense_map`** file could be obtained by [HowNet](https://github.com/thunlp/OpenHowNet-API).



## Data Preparation

You can download the pre-trained character embeddings **`vec.txt`**, pre-trained word-sense embeddings **`sense.txt`** and word-sense map **`sense_map.txt`** from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/5fe72c2b6af6453b8441/) or [Google Drive](https://drive.google.com/open?id=1KQZTyeN2m-5Xmr1QRxuy-7zv2Pzo3so4). Then put them in place following the folder structure:

```
MG-Lattice
|-- ...
|-- data
	|
	|-- sense.txt
	|
	|-- vec.txt
	|
	|-- sense_map.txt
	|
	|-- DATASET_NAME_1
	|	|
	|	|-- train.txt
	|	|-- valid.txt
	|	|-- test.txt
	|	|-- relation2id.txt
    	|
   	|-- DATASET_NAME_2
    		|-- ...
```



## How to run

Arguments are set in `configure.py`, the default values are for **SanWen** dataset. The full usage is:

```tex
-- savemodel  			path to save the model					
-- loadmodel			path to load the model					
-- savedset			path to load the data settings 			

-- public_path			the parent path of the dataset 			(data/)
-- dataset          		the folder name of dataset			(SanWen/)
-- train_file			train dataset  					(train.txt)
-- dev_file			developement dataset  				(dev.txt)
-- test_file			test dataset  					(test.txt)
-- relation2id			map relation to id  				(relation2id.txt)
-- char_emb_file		pre-trained char embeddings 			(vec.txt)
-- sense_emb_file		pre-trained sense embeddings 			(sense.txt)
-- word_sense_map		record polysemous words 			(sense_map.txt)
-- max_length			the max length of the input				
					
-- Encoder			Specify which encoder to use
-- Optimizer			Specify which optimizier to use
-- lr				learning rate							
-- weights_mode			mode to set weights for each class in loss function
```

 With appropriate configuration and data preparation, you can run the model by:

```shell
python main.py
```



## Citation

If you use the code, please cite the paper:

> ```latex
> @inproceedings{li2019chinese,
>   title={Chinese Relation Extraction with Multi-Grained Information andExternal Linguistic Knowledge},
>   author={Li, Ziran and Ding, Ning and Liu, Zhiyuan and Zheng, Hai-Tao and Shen, Ying},
>   booktitle={Proceedings of ACL 2019},
>   year={2019}
> }
> ```





