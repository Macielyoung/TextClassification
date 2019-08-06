# coding:utf-8

import torch
import numpy as np
import random
from config import Config

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.vocab = []
        self.vocab_num_dict = {}
        self.num_vocab_dict = {}
        self.slot_num_dict = {}
        self.num_slot_dict = {}
        self.action_num_dict = {}
        self.num_action_dict = {}

    def get_vocab_and_dict(self, lines):
        self.vocab.append("#pad")
        self.vocab_num_dict['#pad'] = 0
        self.num_vocab_dict[0] = '#pad'
        word_id = 1
        for line in lines:
            arr = [element for element in line.strip().split("|") if element != ""]
            if len(arr) < 3:
                arr.append("None")
            sentence, action, slots = arr
            word_list = [word for word in sentence.split(" ") if word != ""]
            for word in word_list:
                if word not in self.vocab:
                    self.vocab.append(word)
                    self.num_vocab_dict[word_id] = word
                    self.vocab_num_dict[word] = word_id
            if slots == "":
                slots = "None"
            slot_list = [slot for slot in slots.split(";")]
            action_id, slot_id = 0, 0
            if action not in self.action_num_dict:
                self.action_num_dict[action] = action_id
                self.num_action_dict[action_id] = action
                action_id += 1
            for slot in slot_list:
                if slot not in self.slot_num_dict:
                    self.slot_num_dict[slot] = slot_id
                    self.num_slot_dict[slot_id] = slot
                    slot_id += 1
    
    def parse_sentence(self, arr):
        # 获取句子表达
        word_list = [word for word in arr[0].split(" ") if word != ""]
        word_id_list = [self.vocab_num_dict[word] for word in word_list]
        # padding或者句子切割，使得句子长度一致
        if len(word_list) < self.config.max_sen_len:
            word_id_list += [0] * (self.config.max_sen_len - len(word_list))
        else:
            word_id_list = word_list[:self.config.max_sen_len]
        return word_id_list

    def parse_action(self, arr):
        # 获取句子的action label，多分类问题
        action_label = self.action_num_dict[arr[1]] + 1
        return action_label

    def parse_slots(self, arr):
        # 获取句子的slot label(slot 数量不确定，可能是一个也可能是多个)，多分类问题
        slot_label = [0] * len(self.slot_num_dict)
        for slot in arr[2].split(";"):
            slot_label[self.slot_num_dict[slot]] = 1
        return slot_label

    def read_lines(self, datafile):
        with open(datafile, 'r') as f:
            lines = f.readlines()
        return lines

    def get_data(self, lines):
        triples = []
        self.get_vocab_and_dict(lines)
        for line in lines:
            arr = [element for element in line.strip().split("|") if element != ""]
            if len(arr) < 3:
                arr.append("None")
            word_list = self.parse_sentence(arr)
            action_label = self.parse_action(arr)
            slot_label = self.parse_slots(arr)
            triple = {
                'sentence': word_list, 
                'action': action_label,
                'slot': slot_label
            }
            triples.append(triple)
        print("Loaded sentence examples into triples.")
        return triples

    def get_action_batches(self, triples, batches_num):
        sentence_batches, action_batches = [], []
        for _ in range(batches_num):
            triple_batch = random.sample(triples, self.config.batch_size)
            sentence_batch, action_batch = [], []
            for batch in triple_batch:
                sentence_batch.append(batch['sentence'])
                action_batch.append(batch['action'])
            sentence_batches.append(sentence_batch)
            action_batches.append(action_batch)
        return sentence_batches, action_batches

    def get_slot_batches(self, triples, batches_num):
        sentence_batches, slot_batches = [], []
        for _ in range(batches_num):
            triple_batch = random.sample(triples, self.config.batch_size)
            sentence_batch, slot_batch = [], []
            for batch in triple_batch:
                sentence_batch.append(batch['sentence'])
                slot_batch.append(batch['slot'])
            sentence_batches.append(sentence_batch)
            slot_batches.append(slot_batch)
        return sentence_batches, slot_batches

if __name__ == "__main__":
    config = Config()
    train_file = "data_labeled2"
    dataset = Dataset(config)
    lines = dataset.read_lines(train_file)
    triples = dataset.get_data(lines)
    print(triples[0])