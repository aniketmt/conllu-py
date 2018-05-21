# import xml.etree.ElementTree as ET
import numpy as np


class Data(object):
    """Class to get data in required format"""

    def __init__(self, data_file='UD_English-EWT/'):
        self.data_file = data_file
        # self.data_file = [train_file, test_file, dev_file]

        self.vocab = []
        self.postags = []
        self.deprels = []
        # self.init_lists()

    # def init_lists(self):
    #     root = ET.parse(self.data_dir + 'stats.xml').getroot()

    #     for child in root:
    #         if child.tag == 'tags':
    #             for gchild in child:
    #                 self.postags.append(gchild.attrib['name'])
    #         elif child.tag == 'deps':
    #             for gchild in child:
    #                 self.deps.append(gchild.attrib['name'])
    #     self.postags.append('_')
    #     self.deps.append('_')

    def get_file_data(self):
        data_list = []

        with open(self.data_file, 'r') as fdata:
            ddicts, dtree = [], []
            for line in fdata:
                if line == '\n':
                    data_list.append([ddicts, dtree])
                    ddicts, dtree = [], []
                else:
                    attr = line.split('\t')
                    if len(attr) > 1:
                        # Update lists
                        if attr[2] not in self.vocab:
                            self.vocab.append(attr[2])
                        if attr[3] not in self.postags:
                            self.postags.append(attr[3])
                        if attr[7] not in self.deprels:
                            self.deprels.append(attr[7])
                        
                        # Make dictionary
                        ddicts.append({'id': attr[0], 'word': self.vocab.index(attr[2]),
                                       'pos': self.postags.index(attr[3]), 'deprel': self.deps.index(attr[7])})
                        # Generate tree
                        edge = {'head': attr[6], 'deprel': self.deps.index(attr[7]), 'dep': attr[0]}
                        if edge not in dtree:
                            dtree.append(edge)

        return data_list

    def get_transitions(self, word_list, edge_list):
        # print('word list {0}, edge_list {1}'.format(len(word_list), len(edge_list)))
        sigma = ['0']
        beta = [word['id'] for word in word_list]

        heads = {edge['dep']: edge['head'] for edge in edge_list}
        rels = {edge['dep']: edge['deprel'] for edge in edge_list}  # to check right-arc condition

        cedges = []              # list of edges for config
        configs, trans = [], []  # 0 for shift, 1 for left-arc, 2 for right arc
        count = 0

        while len(beta) > 0:
            configs.append([sigma, beta, cedges])

            if len(sigma) == 1:
                trans.append((0, None))
                sigma.insert(0, beta.pop(0))

            elif sigma[0] in heads and heads[sigma[0]] == beta[0]:
                trans.append((1, rels[sigma[0]]))
                heads.pop(sigma[0], None)
                cedges.append([beta[0], rels[sigma[0]], sigma[0]])
                sigma.pop(0)

            else:
                head_flag = False
                for key in heads:
                    if heads[key] == beta[0]:
                        head_flag = True
                        break
                if head_flag == False and heads[beta[0]] == sigma[0]:
                    trans.append((2, rels[beta[0]]))
                    heads.pop(beta[0], None)
                    cedges.append([sigma[0], rels[beta[0]], beta[0]])
                    beta.pop(0)
                    beta.insert(0, sigma.pop(0))
                else:
                    trans.append((0, None))
                    sigma.insert(0, beta.pop(0))
            count += 1

        configs.append([sigma, beta, cedges])
        trans.append((2, self.deps.index('root')))

        if len(cedges) < len(edge_list) - 1:
            configs, trans = [], []

        return np.array(configs), np.array(trans)

    def get_data(self, train=0):
        train_list = self.get_file_data(self.data_file[train])
        train_data = []
        for sentence in train_list:
            word_list, edge_list = sentence
            configs, trans = self.get_transitions(word_list, edge_list)
            train_data.append([word_list, configs, trans])
        return train_data


def test_Data():
    data = Data()
    data_list = data.get_data()

    print(data_list[49][1][0])
    print(data_list[49][2][0])
    print('\n', data_list[49][1][-1])
    print(data_list[49][2][-1])


# test_Data()
