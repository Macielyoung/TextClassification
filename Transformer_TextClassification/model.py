# coding:utf-8

import torch
from torch import nn
from copy import deepcopy
from utils import Embedding, PositionalEncoding
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from ffn import PositionwiseFeedForward
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_model(model, sen_batches, ac_batches):
    all_preds, all_y = [], []
    for ids, sen_batch in enumerate(sen_batches):
        if torch.cuda.is_available():
            x = torch.Tensor(sen_batch).cuda()
        else:
            x = torch.Tensor(sen_batch)
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y += ac_batches[ids]
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score

# transformer model
class Transformer(nn.Module):
    def __init__(self, config, src_vocab):
        super(Transformer, self).__init__()
        self.config = config
        self.src_vocab = src_vocab
        
        # 超参数
        # h是多头数量， N是层数， dropout是比率
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        # 词向量维度，全连接维度
        d_model, d_ff = self.config.d_model, self.config.d_ff

        # 多头注意力层
        attn = MultiHeadedAttention(h, d_model)
        # 全连接层
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 位置向量
        position = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embedding(self.config.d_model, self.src_vocab), deepcopy(position)) # embedding with position encoding

        self.fc = nn.Linear(self.config.d_model, self.config.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sentences = self.src_embed(x) # shape = (batch_size, sen_len, d_model)
        encoded_sentences = self.encoder(embedded_sentences)

        # Convert input to (batch_size, d_model) for linear layer
        final_feature_map = encoded_sentences[:,-1,:]
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, sentence_batches, action_batches, epoch):
        train_losses = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()
        
        batches_num = range(len(sentence_batches))
        for sentence_batch, action_batch, i in zip(sentence_batches, action_batches, batches_num):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = torch.LongTensor(sentence_batch).cuda()
                y = torch.LongTensor(action_batch).cuda()
            else:
                x = torch.LongTensor(sentence_batch)
                y = torch.LongTensor(action_batch)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # # Evalute Accuracy on validation set
                # val_accuracy = evaluate_model(self, val_iterator)
                # print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                # self.train()
        return train_losses