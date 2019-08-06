# coding:utf-8

from data_loader import Dataset
from model import Transformer, evaluate_model
from torch import nn
import sys
from config import Config
import torch.optim as optim
import torch

if __name__ == "__main__":
    config = Config()
    train_file = "data_labeled"
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    
    dataset = Dataset(config)
    lines = dataset.read_lines(train_file)
    triples = dataset.get_data(lines)
    sentence_batches, action_batches = dataset.get_action_batches(triples, config.batches_num)

    model = Transformer(config, len(dataset.vocab))
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # 多分类交叉熵损失函数
    NLLLoss = nn.NLLLoss()
    # 模型加载优化函数和损失函数
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)

    train_losses = []
    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss = model.run_epoch(sentence_batches, action_batches, i)
        train_losses.append(train_loss)

    train_acc = evaluate_model(model, sentence_batches, action_batches)
    print("Final Training Dataset Accuracy: {.4f}".format(train_acc))