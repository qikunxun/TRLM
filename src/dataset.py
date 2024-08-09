import os
import json
import numpy as np
from copy import deepcopy

def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x

class Triple:
    def __init__(self, h, r, t, time):
        self.h = h
        self.r = r
        self.t = t
        self.time = time

    def get_triple(self):
        return self.h, self.r, self.t, self.time

class Dataset:

    def __init__(self, kg_path, batch_size, target_relation, option=None, max_size=500):
        self.option = option
        self.batch_size = batch_size
        self.target_relation = target_relation
        self.kg, self.kg_valid, self.kg_test, self.id2entity, self.entity2id, self.id2relation, \
                self.relation2id, self.time2id, self.id2time, self.triple2id, self.id2triple, self.train_kg, \
                self.time2tripleid, self.tripleid2time, self.triple2id_valid, self.time2tripleid_valid, \
                self.tripleid2time_valid, self.triple2id_test, self.time2tripleid_test, \
                self.tripleid2time_test = self.load_kg_all(kg_path)
        self.graph, self.graph_entity, self.relation_tail, self.relation_head = self.build_graph()
        self.targets_h = self.get_targets_head()
        self.targets_t = self.get_targets_tail()
        self.max_size = max_size

    def load_kg_all(self, kg_path):
        train_path = os.path.join(kg_path, 'train.txt')
        valid_path = os.path.join(kg_path, 'valid.txt')
        test_path = os.path.join(kg_path, 'test.txt')
        entity_path = os.path.join(kg_path, 'entity2id.json')
        relation_path = os.path.join(kg_path, 'relation2id.json')
        time_path = os.path.join(kg_path, 'ts2id.json')

        with open(entity_path, mode='r') as fd:
            entity2id = json.load(fd)
        id2entity = reverse(entity2id)

        with open(relation_path, mode='r') as fd:
            relation2id = json.load(fd)
        id2relation = reverse(relation2id)
        length = len(relation2id)
        for i in range(length):
            relation2id['INV' + id2relation[i]] = len(relation2id)
            id2relation[len(relation2id) - 1] = 'INV' + id2relation[i]
        relation2id['Identity'] = len(relation2id)
        id2relation[len(id2relation)] = 'Identity'

        with open(time_path, mode='r') as fd:
            time2id = json.load(fd)
        time2id['AnyTime'] = -1
        id2time = reverse(time2id)

        train_triples = []
        triple2id = {}
        kg = []
        kg_valid = []
        kg_test = []
        with open(train_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 4: continue
                h, r, t, cur_time = items
                triple = Triple(h, r, t, cur_time)
                triple_inv = Triple(t, 'INV' + r, h, cur_time)
                kg.append(triple)
                kg.append(triple_inv)
                train_triples.append(triple)
                train_triples.append(triple_inv)
                triple_index = '{}\t{}\t{}\t{}'.format(h, r, t, cur_time)
                triple_index_inv = '{}\t{}\t{}\t{}'.format(t, 'INV' + r, h, cur_time)
                if triple not in triple2id: triple2id[triple_index] = time2id[cur_time]
                if triple_inv not in triple2id: triple2id[triple_index_inv] = time2id[cur_time]
                # if len(kg) > 2000: break
        triple2id_valid = deepcopy(triple2id)
        
        with open(valid_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 4: continue
                h, r, t, cur_time = items
                triple = Triple(h, r, t, cur_time)
                triple_inv = Triple(t, 'INV' + r, h, cur_time)
                kg_valid.append(triple)
                kg_valid.append(triple_inv)
                triple_index = '{}\t{}\t{}\t{}'.format(h, r, t, cur_time)
                triple_index_inv = '{}\t{}\t{}\t{}'.format(t, 'INV' + r, h, cur_time)
                if triple not in triple2id_valid: triple2id_valid[triple_index] = time2id[cur_time]
                if triple_inv not in triple2id_valid: triple2id_valid[triple_index_inv] = time2id[cur_time]

        triple2id_test = deepcopy(triple2id_valid)
        
        with open(test_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 4: continue
                h, r, t, cur_time = items
                triple = Triple(h, r, t, cur_time)
                triple_inv = Triple(t, 'INV' + r, h, cur_time)
                kg_test.append(triple)
                kg_test.append(triple_inv)
                triple_index = '{}\t{}\t{}\t{}'.format(h, r, t, cur_time)
                triple_index_inv = '{}\t{}\t{}\t{}'.format(t, 'INV' + r, h, cur_time)
                if triple not in triple2id_test: triple2id_test[triple_index] = time2id[cur_time]
                if triple_inv not in triple2id_test: triple2id_test[triple_index_inv] = time2id[cur_time]
                
        for entity in entity2id:
            triple = Triple(entity, 'Identity', entity, 'AnyTime')
            triple_index = '{}\t{}\t{}\t{}'.format(entity, 'Identity', entity, 'AnyTime')
            triple2id[triple_index] = -1
            triple2id_valid[triple_index] = -1
            triple2id_test[triple_index] = -1
            kg.append(triple)
            kg_valid.append(triple)
            kg_test.append(triple)

        sorted_id = sorted(triple2id.items(), key=lambda x: x[1])
        time2tripleid = {}
        tripleid2time = {}
        for i, item in enumerate(sorted_id):
            triple2id[item[0]] = i
            time_id = item[1] + 1
            if time_id not in time2tripleid: time2tripleid[time_id] = i
            tripleid2time[i] = item[1]
        id2triple = reverse(triple2id)

        sorted_id_valid = sorted(triple2id_valid.items(), key=lambda x: x[1])
        time2tripleid_valid = {}
        tripleid2time_valid = {}
        for i, item in enumerate(sorted_id_valid):
            triple2id_valid[item[0]] = i
            time_id = item[1] + 1
            if time_id not in time2tripleid_valid: time2tripleid_valid[time_id] = i
            tripleid2time_valid[i] = item[1]
        
        sorted_id_test = sorted(triple2id_test.items(), key=lambda x: x[1])
        time2tripleid_test = {}
        tripleid2time_test = {}
        for i, item in enumerate(sorted_id_test):
            triple2id_test[item[0]] = i
            time_id = item[1] + 1
            if time_id not in time2tripleid_test: time2tripleid_test[time_id] = i
            tripleid2time_test[i] = item[1]
        print(len(triple2id), len(triple2id_valid), len(triple2id_test))
        return kg, kg_valid, kg_test, id2entity, entity2id, id2relation, relation2id, time2id, id2time, triple2id, \
               id2triple, train_triples, time2tripleid, tripleid2time, triple2id_valid, \
               time2tripleid_valid, tripleid2time_valid, triple2id_test, time2tripleid_test, tripleid2time_test

    def batch_iter(self):
        sampled_data = np.random.choice(self.train_kg, size=self.option.iteration_per_batch * self.batch_size)
        num_batch = len(sampled_data) // self.batch_size
        for i in range(num_batch):
            batch_ori = sampled_data[i * self.batch_size: (i + 1) * self.batch_size]
            triples = []
            for triple_batch in batch_ori:
                h, r, t, time = triple_batch.get_triple()
                triples.append((h, r, t, time))
            yield triples


    def build_graph(self):
        graph = {}
        graph_entity = {}
        relation_head = {}
        relation_tail = {}
        for triple in self.kg:
            h, r, t, time = triple.get_triple()
            if h not in graph:
                graph[h] = {r: [[t, time]]}
                graph_entity[h] = {t: [[r, time]]}
            else:
                if r in graph[h]:
                    graph[h][r].append([t, time])
                else:
                    graph[h][r] = [[t, time]]

                if t in graph_entity[h]:
                    graph_entity[h][t].append([r, time])
                else:
                    graph_entity[h][t] = [[r, time]]
            if r == self.target_relation and t not in relation_tail: relation_tail[t] = len(relation_tail)
            if r == self.target_relation and h not in relation_head: relation_head[h] = len(relation_head)
        return graph, graph_entity, relation_tail, relation_head

    def get_targets_tail(self):
        targets = []
        relation_tail = sorted(self.relation_tail.items(), key=lambda d: d[1])
        for item in relation_tail:
            targets.append(item[0])
        return targets

    def get_targets_head(self):
        targets = []
        entities = sorted(self.entity2id.items(), key=lambda d: d[1])
        for item in entities:
            targets.append(item[0])
        return targets

    def select_random_entity_tail(self, targets, cur, h, max_deep=5):
        selected_entity = None
        count = 0
        while selected_entity is None or selected_entity == cur or \
                (h in self.graph and self.target_relation in self.graph[h]
                 and selected_entity in self.graph[h][self.target_relation]):
            if count == max_deep: break
            selected_entity = np.random.choice(targets, 1)[0]
            count += 1
        return selected_entity

    def select_random_entity_head(self, targets, cur, t, max_deep=5):
        selected_entity = None
        count = 0
        while selected_entity is None or selected_entity == cur or \
                (selected_entity in self.graph and self.target_relation in self.graph[selected_entity]
                 and t in self.graph[selected_entity][self.target_relation]):
            # if count == max_deep: break
            selected_entity = np.random.choice(targets, 1)[0]
            count += 1
        return selected_entity

