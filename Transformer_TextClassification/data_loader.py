# coding:utf-8

import torch
import numpy as np

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
        self.slots = []
        self.actions = []
        self.slot_num_dict = {}
        self.num_slot_dict = {}
        self.action_num_dict = {}
        self.num_action_dict = {}

    def get_vocab_and_dict(self, lines):
        self.vocab.append("#pad")
        for line in lines:
            arr = [element for element in line.strip().split("|") if element != ""]
            sentence, action, slots = arr
            word_list = [word for word in sentence.split(" ") if word != ""]
            for word in word_list:
                if word not in self.vocab:
                    self.vocab.append(word)
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
        # padding或者句子切割，使得句子长度一致
        if len(word_list) < self.config.max_sen_len:
            word_list += ["#pad"] * (self.config.max_sen_len - len(word_list))
        else:
            word_list = word_list[:self.config.max_sen_len]
        return word_list

    def parse_action(self, arr):
        # 获取句子的action label，多分类问题
        action_label = [0] * len(self.action_num_dict)
        action_label[self.action_num_dict[arr[1]]] = 1
        return action_label

    def parse_slots(self, arr):
        # 获取句子的slot label(slot 数量不确定，可能是一个也可能是多个)，多分类问题
        slots = "None" if arr[2] == "" else arr[2]
        slot_label = [0] * len(self.slot_num_dict)
        for slot in slots.split(";"):
            slot_label[self.slot_num_dict[slot]] = 1
        return slot_label

    def read_data(self, datafile):
        sen_lines = []
        with open(datafile, 'r') as f:
            lines = f.readlines()
            self.get_vocab_and_dict(lines)
            for line in lines:
                arr = [element for element in line.strip().split("|") if element != ""]
                word_list = self.parse_sentence(arr)
                action_label = self.parse_action(arr)
                slot_label = self.parse_slots(arr)
                sentence_label = word_list + action_label + slot_label
                sen_lines.append(sentence_label)
        return sen_lines