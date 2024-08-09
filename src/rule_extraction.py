import os
import json
import sys
import torch
from tqdm import tqdm

beam_size = int(sys.argv[4])

class Option(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)


def load_kg(kg_file):
    kg = []
    with open(kg_file, mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            if len(items) != 3: continue
            h, r, t = items
            kg.append((h, r, t))
    return kg

def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x

def build_graph(kg, target_relation):
    graph = {}
    graph_entity = {}
    for triple in kg:
        h, r, t, c = triple.get_triple()
        if h not in graph:
            graph[h] = {r: [t]}
            graph_entity[h] = {t: [r]}
        else:
            if r in graph[h]:
                graph[h][r].append(t)
            else:
                graph[h][r] = [t]

            if t in graph_entity[h]:
                graph_entity[h][t].append(r)
            else:
                graph_entity[h][t] = [r]
    return graph, graph_entity

def init_matrix(matrix, kg, entity2id, entity2id_tail, relation2id):
    print('Processing Matirx(shape={})'.format(matrix.shape))
    for triple in tqdm(kg):
        h, r, t = triple.get_triple()
        # if r == target_relation: continue
        if t not in entity2id_tail: continue
        entity_a = entity2id[h]
        entity_b = entity2id_tail[t]
        relation = relation2id[r]
        matrix[entity_a][entity_b][relation] = 1

def get_head(heads, kg):
    entity2id_head = {}
    id2entity_head = {}
    for h in heads:
        entity2id_head[h] = len(entity2id_head)
        id2entity_head[entity2id_head[h]] = h
    for h, r, t in kg:
        if h not in entity2id_head: continue
        if t in entity2id_head: continue
        entity2id_head[t] = len(entity2id_head)
        id2entity_head[entity2id_head[t]] = t
    return entity2id_head, id2entity_head


def get_beam(indices, t, beam_size, r, T, all_indices):
    beams = indices[:, :beam_size]
    for l in range(indices.shape[0]):
        index = indices[l]
        j = 0
        for i in range(index.shape[-1]):
            beams[l][j] = index[i]
            j += 1
            if j == beam_size: break

    return beams

def get_states(indices, scores):
    states = torch.zeros(indices.shape)
    for l in range(indices.shape[0]):
        states[l] = torch.index_select(scores[l], -1, indices[l])
    return states

def transform_score(x, T):
    one = torch.autograd.Variable(torch.Tensor([1]))
    zero = torch.autograd.Variable(torch.Tensor([0]).detach())
    return torch.minimum(torch.maximum(x / T, zero), one)

def analysis(id2relation, relation2id, option, target_relation, model_save_path, inverse=False):
    ckpt = torch.load(model_save_path, map_location=torch.device('cpu'))
    r = len(relation2id)
    T = option.step
    w_ = torch.log_softmax(ckpt['w.{}'.format(relation2id[target_relation])][0], dim=-1)
    scores = w_
    indices_order = torch.argsort(scores, dim=-1, descending=True)
    all_indices = []
    indices = get_beam(indices_order, T - 1, beam_size, len(relation2id), T, all_indices)
    states = get_states(indices, scores)

    all_indices.append(indices)
    for t in range(1, T):
        w_ = torch.log_softmax(ckpt['w.{}'.format(relation2id[target_relation])][t], dim=-1)
        scores = states.unsqueeze(dim=-1) + w_.unsqueeze(dim=1)
        scores = scores.view(option.length, -1)
        indices_order = torch.argsort(scores, dim=-1, descending=True)
        topk_indices = get_beam(indices_order, t, beam_size, len(relation2id), T, all_indices)
        states = get_states(topk_indices, scores)
        all_indices.append(topk_indices)
    outputs = torch.zeros(option.length, option.step, beam_size).long()
    p = torch.zeros(option.length, beam_size).long()
    for beam in range(beam_size):
        p[:, beam] = beam
    for t in range(option.step):
        for l in range(option.length):
            for beam in range(beam_size):
                c = int(all_indices[t][l][p[l][beam]] % (r + 1))
                outputs[l][t][beam] = c
                p_new = int(all_indices[t][l][p[l][beam]] / (r + 1))
                p[l][beam] = p_new

    all_rules = []
    for l in range(option.length):
        if inverse:
            head = 'INV' + target_relation
        else:
            head = target_relation
        rule = '{}(X, Y, T)<-'.format(head)
        rules = [rule] * beam_size
        counts = torch.zeros(option.length, beam_size)
        for t in range(option.step):
            for beam in range(beam_size):
                c = int(outputs[l][t][beam])
                if c < r:
                    tmp = id2relation[c]
                    x = 'X'
                    if counts[l][beam] > 0: x = 'Z_{}'.format(int(counts[l][beam]) - 1)
                    y = 'Z_{}'.format(int(counts[l][beam]))
                    u = 'T_{}'.format(int(counts[l][beam]))
                    # if t == option.step - 1 or (t > 0 and outputs[l][t - 1][beam] == r): y = 'Y'
                    flag = tmp + '({}, {}, {})'.format(x, y, u)
                    counts[l][beam] += 1
                    end = ''
                    if t != 0 and not rules[beam].endswith('<-'): end = ' ∧ '
                    rules[beam] += end + flag
        # print(rules, float(model.weight[l]))
        rules_new = []
        for beam in range(beam_size):
            and_count = rules[beam].count(' ∧ ')
            rules[beam] = rules[beam].replace('Z_{}'.format(and_count), 'Y')
            constraints = ['']
            for k in range(and_count + 1):
                if k < and_count:
                    term = '_{}'.format(k + 1)
                    constraints.append('≤(T_{}, T{})'.format(k, term))
                if k == and_count - 1: constraints.append('<(T_{}, T)'.format(k + 1))
            if '<-' in rules[beam]: rules_new.append(rules[beam] + ' ∧ '.join(constraints))
        all_rules.append(rules_new)

    ids_sort = torch.argsort(ckpt['weight.{}'.format(relation2id[target_relation])].squeeze(dim=-1), descending=True)
    fw = open('./{}/rules-{}.txt'.format(option.exps_dir, target_relation), mode='w')
    for i, ids in enumerate(ids_sort):
        weight = torch.sigmoid(ckpt['weight.{}'.format(relation2id[target_relation])][int(ids)])
        data = {'rank': (i + 1), 'id': int(ids), 'rules': all_rules[int(ids)], 'weight': float(weight)}
        fw.write(json.dumps(data) + '\n')
        print('Rank: {}, id: {}, Rule: {}, Weight: {}'.format((i + 1), ids, all_rules[int(ids)], float(weight)))
        # break
    fw.close()



if __name__ == '__main__':
    dataset = sys.argv[1]
    exps_dir = sys.argv[2]
    exp_name = sys.argv[3]

    with open('{}/relation2id.json'.format(dataset), mode='r') as fd:
        relation2id = json.load(fd)
    id2relation = reverse(relation2id)
    length = len(relation2id)
    for i in range(length):
        relation2id['INV' + id2relation[i]] = len(relation2id)
        id2relation[len(relation2id) - 1] = 'INV' + id2relation[i]

    for target_relation in relation2id:
        option = Option('./{}/{}/'.format(exps_dir, exp_name))
        analysis(id2relation, relation2id, option, target_relation, '{}/model.pt'.format(option.exp_dir), False)