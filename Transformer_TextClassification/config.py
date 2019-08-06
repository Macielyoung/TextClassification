# coding:utf-8

class Config(object):
    N = 1 #编码器中子层数量，transformer论文中使用6个
    d_model = 256 #模型维数，transformer中使用512
    d_ff = 512 
    h = 8
    dropout = 0.1
    output_size = 20
    lr = 0.0003
    max_epochs = 50
    batch_size = 128
    max_sen_len = 30
    batches_num = 1000