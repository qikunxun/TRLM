import argparse
import time
import random
import json
import pickle as pkl
import torch
import os
import numpy as np
from copy import deepcopy
from model import TRLMModel
from tqdm import tqdm
from dataset import Dataset

class Option(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        if not os.path.exists(self.exps_dir):
            os.mkdir(self.exps_dir)

        self.exp_dir = os.path.join(self.exps_dir, self.exp_name)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        with open(os.path.join(self.exp_dir, "option.txt"), "w") as f:
            json.dump(self.__dict__, f, indent=1)
        return True


def set_seed(option):
    random.seed(option.seed)
    np.random.seed(option.seed)
    torch.manual_seed(option.seed)
    os.environ['PYTHONHASHSEED'] = str(option.seed)
    if option.use_gpu: torch.cuda.manual_seed_all(option.seed)


def save_data(kg, entity2id, relation2id, file_path=None):
    with open(os.path.join(file_path, 'kg.pkl'), mode='wb') as fw:
        pkl.dump(kg, fw)
    with open(os.path.join(file_path, 'entity2id.pkl'), mode='wb') as fw:
        pkl.dump(entity2id, fw)
    with open(os.path.join(file_path, 'relation2id.pkl'), mode='wb') as fw:
        pkl.dump(relation2id, fw)



def get_indices(matrix_all):
    indices_all = {}
    indices = matrix_all.indices()
    for i in range(indices.shape[-1]):
        index = indices[:, i]
        flag = '{}\t{}'.format(index[0], index[1])
        indices_all[flag] = i
    return indices_all


def mask_data(matrix, triples, indices, entity2id, triple2id, score, dim):
    values = matrix.values()
    for h, r, t, cur_time in triples:
        triple = '{}\t{}\t{}\t{}'.format(h, r, t, cur_time)
        triple_id = triple2id[triple]
        if dim == 0:
            flag = '{}\t{}'.format(triple_id, entity2id[t])
        else:
            flag = '{}\t{}'.format(entity2id[h], triple_id)
        index = indices[flag]
        values[index] = score


def split_data(kg):
    data_ori = {}
    data_inv = {}
    for triple in kg:
        h, r, t, cur_time = triple.get_triple()
        if 'INV' not in r:
            if r in data_ori:
                data_ori[r].append((h, r, t, cur_time))
            else:
                data_ori[r] = [(h, r, t, cur_time)]
        else:
            if r in data_inv:
                data_inv[r].append((h, r, t, cur_time))
            else:
                data_inv[r] = [(h, r, t, cur_time)]
    return data_ori, data_inv


def extend_graph(graph_entity, valid_data):
    for triple in valid_data:
        h, r, t, cur_time = triple.get_triple()
        if h not in graph_entity:
            graph_entity[h] = {t: [[r, cur_time]]}
        else:
            if t in graph_entity[h]:
                graph_entity[h][t].append([r, cur_time])
            else:
                graph_entity[h][t] = [[r, cur_time]]



def create_graph(kg, entity2id, relation2id, time2id, triple2id, tripleid2time, use_gpu):
    i_x_h = []
    i_y_h = []
    v_h = []
    i_x_t = []
    i_y_t = []
    v_t = []
    i_x_r = []
    i_y_r = []
    v_r = []
    i_x_time = []
    i_y_time = []
    v_time = []
    for triple_ in kg:
        x, r, y, cur_time = triple_.get_triple()
        triple = '{}\t{}\t{}\t{}'.format(x, r, y, cur_time)
        relation = relation2id[r]
        time_id = time2id[cur_time] + 1
        i_x_r.append(triple2id[triple])
        i_y_r.append(relation)
        v_r.append(1)
        i_x_time.append(triple2id[triple])
        i_y_time.append(time_id)
        v_time.append(1)
        i_x_h.append(entity2id[x])
        i_y_h.append(triple2id[triple])
        v_h.append(1)
        i_x_t.append(triple2id[triple])
        i_y_t.append(entity2id[y])
        v_t.append(1)

    i = torch.LongTensor([i_x_h, i_y_h])
    v = torch.FloatTensor(v_h)
    e2triple = torch.sparse.FloatTensor(i, v, torch.Size([len(entity2id), len(triple2id)]))

    i = torch.LongTensor([i_x_t, i_y_t])
    v = torch.FloatTensor(v_t)
    triple2e = torch.sparse.FloatTensor(i, v, torch.Size([len(triple2id), len(entity2id)]))

    i = torch.LongTensor([i_x_r, i_y_r])
    v = torch.FloatTensor(v_r)
    triple2r = torch.sparse.FloatTensor(i, v, torch.Size([len(triple2id), len(relation2id)]))

    i = torch.LongTensor([i_x_time, i_y_time])
    v = torch.FloatTensor(v_time)
    triple2time = torch.sparse.FloatTensor(i, v, torch.Size([len(triple2id), len(time2id)]))

    if use_gpu:
        e2triple = e2triple.cuda()
        triple2e = triple2e.cuda()
        triple2r = triple2r.cuda()
        triple2time = triple2time.cuda()

    return e2triple.coalesce(), triple2e.coalesce(), triple2r.coalesce(), triple2time.coalesce()


def valid_process(valid_all_data, dataset, model, flag, batch_size, e2triple, triple2e, triple2r, triple2time,
                  indices_e2triple, indices_triple2e, graph_entity, triple2id, time2tripleid, option, raw=False):
    mrr = 0
    hit_1 = 0
    hit_3 = 0
    hit_10 = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for target_relation in tqdm(dataset.relation2id):
            if target_relation == 'Identity': continue
            if (flag and 'INV' not in target_relation) or (not flag and 'INV' in target_relation): continue
            if target_relation not in valid_all_data: continue
            valid_data = valid_all_data[target_relation]
            if len(valid_data) == 0: continue

            if len(valid_data) % batch_size == 0:
                batch_num = int(len(valid_data) / batch_size)
            else:
                batch_num = int(len(valid_data) / batch_size) + 1
            for i in range(batch_num):
                triples = valid_data[i * batch_size: (i + 1) * batch_size]
                input_x = []
                input_r = []
                input_triple2id = []
                for x, r, y, cur_time in triples:
                    input_x.append(dataset.entity2id[x])
                    input_r.append(dataset.relation2id[r])
                    end = dataset.time2id[cur_time] + 1
                    start = max(end - option.window, 1)
                    if end in time2tripleid:
                        end = time2tripleid[end]
                    else:
                        while end not in time2tripleid and end < int(triple2time.shape[1]):
                            end += 1
                        if end in time2tripleid:
                            end = time2tripleid[end] - 1
                        else:
                            end = int(triple2time.shape[0]) - 1

                    if start in time2tripleid:
                        start = time2tripleid[start]
                    else:
                        while start not in time2tripleid and start > 0:
                            start -= 1
                        start = time2tripleid[start]
                    input_triple2id.append([start, end])

                input_x = torch.LongTensor(input_x)
                input_r = torch.LongTensor(input_r)
                input_triple2id = torch.LongTensor(input_triple2id)
                if option.use_gpu:
                    input_x = input_x.cuda()
                    input_r = input_r.cuda()
                    input_triple2id = input_triple2id.cuda()
                input_x = torch.nn.functional.one_hot(input_x, len(dataset.entity2id)).float().to_sparse()
                state = model(input_x, input_r, input_triple2id, e2triple, triple2e, triple2r, triple2time,
                                is_training=False, window=option.window)

                state = state.to_dense()
                for i, (x, r, y, cur_time) in enumerate(triples):
                    truth_score_ori = state[i][dataset.entity2id[y]].cpu().clone()
                    scores_head = state[i].cpu().clone()
                    for tail in graph_entity[x]:
                        # if tail not in entity2id: continue
                        r_tmp = r
                        if [r_tmp, cur_time] in graph_entity[x][tail]: scores_head[dataset.entity2id[tail]] = -1e20
                    rank = torch.sum((scores_head > truth_score_ori).int()) + 1
                    mrr += 1 / rank
                    if rank <= 1:
                        hit_1 += 1
                    if rank <= 3:
                        hit_3 += 1
                    if rank <= 10:
                        hit_10 += 1
                    count += 1

        if count > 0:
            mrr /= count
            hit_1 /= count
            hit_3 /= count
            hit_10 /= count
        print('Valid Count:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(count, mrr, hit_1, hit_3, hit_10))
        return mrr, hit_1, hit_3, hit_10


def evaluate(data, dataset, model, e2triple, triple2e, triple2r, triple2time,
             indices_e2triple, indices_triple2e, graph_entity, triple2id, time2tripleid, option, e):

    mrr_ori, hit_1_ori, hit_3_ori, hit_10_ori = valid_process(
        data[0], dataset, model, False, option.batch_size, e2triple, triple2e, triple2r, triple2time,
        indices_e2triple, indices_triple2e, graph_entity, triple2id, time2tripleid, option, raw=False)
    mrr_inv, hit_1_inv, hit_3_inv, hit_10_inv = valid_process(
        data[1], dataset, model, True, option.batch_size, e2triple, triple2e, triple2r, triple2time,
        indices_e2triple, indices_triple2e, graph_entity, triple2id, time2tripleid, option, raw=False)
    mrr = (mrr_ori + mrr_inv) / 2
    hit_1 = (hit_1_ori + hit_1_inv) / 2
    hit_3 = (hit_3_ori + hit_3_inv) / 2
    hit_10 = (hit_10_ori + hit_10_inv) / 2
    print('Valid Current Score : Epoch:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(e,
                                                                                         mrr,
                                                                                         hit_1,
                                                                                         hit_3,
                                                                                         hit_10))
    return mrr, hit_1, hit_3, hit_10


def main(dataset, option):
    print('Current Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('Entity Num:', len(dataset.entity2id))
    print('Relation Num:', len(dataset.relation2id))
    print('Train KG Size:', len(dataset.train_kg))
    graph_entity = deepcopy(dataset.graph_entity)
    extend_graph(graph_entity, dataset.kg_valid)
    extend_graph(graph_entity, dataset.kg_test)

    kg_valid = deepcopy(dataset.kg)
    kg_valid.extend(dataset.kg_valid)
    kg_test = deepcopy(kg_valid)
    kg_test.extend(dataset.kg_test)
    valid_data = split_data(dataset.kg_valid)
    test_data = split_data(dataset.kg_test)

    model = TRLMModel(len(dataset.relation2id), len(dataset.time2id), option.step, option.length,
                      len(dataset.entity2id), option.tau, option.use_gpu)
    if option.use_gpu: model = model.cuda()
    for parameter in model.parameters():
        print(parameter)

    optimizer = torch.optim.Adam(model.parameters(), lr=option.learning_rate)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=option.learning_rate * 0.8)

    end_flag = False
    max_score = -1
    max_record = {'mrr': 0, 'hit_1': 0, 'hit_3': 0, 'hit_10': 0, 'epoch': 0}
    saved_flag = False
    e2triple, triple2e, triple2r, triple2time = create_graph(dataset.kg, dataset.entity2id,
                        dataset.relation2id, dataset.time2id, dataset.triple2id, dataset.tripleid2time, option.use_gpu)
    indices_e2triple = get_indices(e2triple)
    indices_triple2e = get_indices(triple2e)

    entity2id_cur = dataset.entity2id
    for e in range(option.max_epoch):
        model.train()
        total_loss = 0
        if end_flag: break
        for k, batch in enumerate(dataset.batch_iter()):
            start_time = time.time()
            triples = batch
            mask_data(e2triple, triples, indices_e2triple, dataset.entity2id, dataset.triple2id, 0, 1)
            mask_data(triple2e, triples, indices_triple2e, dataset.entity2id, dataset.triple2id, 0, 0)
            loss = 0
            input_x = []
            input_r = []
            input_triple2id = []
            for x, r, y, cur_time in triples:
                input_x.append(entity2id_cur[x])
                input_r.append(dataset.relation2id[r])
                end = dataset.time2id[cur_time] + 1
                start = max(end - option.window, 1)
                if end in dataset.time2tripleid:
                    end = dataset.time2tripleid[end]
                else:
                    while end not in dataset.time2tripleid and end < int(triple2time.shape[1]):
                        end += 1
                    if end in dataset.time2tripleid:
                        end = dataset.time2tripleid[end] - 1
                    else:
                        end = int(triple2time.shape[0]) - 1

                if start in dataset.time2tripleid:
                    start = dataset.time2tripleid[start]
                else:
                    while start not in dataset.time2tripleid and start > 0:
                        start -= 1
                    start = dataset.time2tripleid[start]
                input_triple2id.append([start, end])

            input_x = torch.LongTensor(input_x)
            input_r = torch.LongTensor(input_r)
            input_triple2id = torch.LongTensor(input_triple2id)
            if option.use_gpu:
                input_x = input_x.cuda()
                input_r = input_r.cuda()
                input_triple2id = input_triple2id.cuda()
            input_x = torch.nn.functional.one_hot(input_x, len(entity2id_cur)).float().to_sparse()
            state = model(input_x, input_r, input_triple2id, e2triple, triple2e, triple2r, triple2time,
                            is_training=True, window=option.window)
            # if state._nnz() == 0: continue
            one_hot = torch.zeros(len(triples), len(entity2id_cur)).to(state.device)
            # logit_mask = torch.zeros(len(triples), len(entity2id_cur)).to(state.device)
            for i, (x, r, y, cur_time) in enumerate(triples):
                logit_mask = torch.zeros(len(entity2id_cur)).to(state.device)
                one_hot[i][entity2id_cur[y]] = 1
                for ent in dataset.graph[x][r]:
                    if ent[0] == y: continue
                    if ent[1] != cur_time: continue
                    logit_mask[entity2id_cur[ent[0]]] = 1
                # score = state.select(0, i).select(0, entity2id_cur[y]).item()
                loss += model.log_loss(torch.unsqueeze(state[i], dim=0),
                                       entity2id_cur[y], logit_mask)

            total_loss += loss.item()
            if loss > option.threshold:
                loss.backward()
                optimizer.step()
                sch.step(loss)
                optimizer.zero_grad()
            mask_data(e2triple, triples, indices_e2triple, dataset.entity2id, dataset.triple2id, 1, 1)
            mask_data(triple2e, triples, indices_triple2e, dataset.entity2id, dataset.triple2id, 1, 0)
            end_time = time.time()
            print('Epoch: {}, Batch: {}, Loss: {}, Time cost: {}'.format(e, k, loss.item(), end_time - start_time))

        print('Epoch: {}, Total Loss: {}'.format(e, total_loss))
        if option.early_stop:
            mrr, hit_1, hit_3, hit_10 = evaluate(valid_data, dataset, model, e2triple, triple2e,
                                            triple2r, triple2time, indices_e2triple,
                                            indices_triple2e, graph_entity, dataset.triple2id,
                                            dataset.time2tripleid, option, e)
            if mrr > max_score:
                torch.save(model.state_dict(),
                           os.path.join(option.exp_dir, 'model.pt'))
                saved_flag = True

    if not saved_flag:
        torch.save(model.state_dict(),
                   os.path.join(option.exp_dir, 'model.pt'))


def test(dataset, option):
    print('Current Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('Entity Num:', len(dataset.entity2id))
    print('Relation Num:', len(dataset.relation2id))
    print('Train KG Size:', len(dataset.train_kg))
    graph_entity = deepcopy(dataset.graph_entity)
    extend_graph(graph_entity, dataset.kg_valid)
    extend_graph(graph_entity, dataset.kg_test)

    kg_valid = deepcopy(dataset.kg)
    kg_valid.extend(dataset.kg_valid)
    kg_test = deepcopy(kg_valid)
    kg_test.extend(dataset.kg_test)
    valid_data = split_data(dataset.kg_valid)
    test_data = split_data(dataset.kg_test)

    model = TRLMModel(len(dataset.relation2id), len(dataset.time2id), option.step, option.length,
                      len(dataset.entity2id), option.tau, option.use_gpu)
    model_save_path = os.path.join(option.exp_dir, 'model.pt')
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    if option.use_gpu: model = model.cuda()
    for parameter in model.parameters():
        print(parameter)
    e2triple, triple2e, triple2r, triple2time = create_graph(dataset.kg, dataset.entity2id,
                        dataset.relation2id, dataset.time2id, dataset.triple2id, dataset.tripleid2time, option.use_gpu)

    mrr, hit_1, hit_3, hit_10 = evaluate(test_data, dataset, model, e2triple, triple2e,
                                    triple2r, triple2time, None, None, graph_entity, dataset.triple2id,
                                    dataset.time2tripleid, option, -1)

    return mrr, hit_1, hit_3, hit_10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--use_gpu', default=False, action="store_true")
    parser.add_argument('--gpu_id', default=4, type=int)
    # model architecture
    parser.add_argument('--length', default=3, type=int)
    parser.add_argument('--step', default=3, type=int)
    parser.add_argument('--window', default=500, type=int)
    parser.add_argument('--tau', default=10, type=float)
    parser.add_argument('--target_relation', default=None, type=str)
    parser.add_argument('--inverse', default=False, action="store_true")
    # optimization
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--iteration_per_batch', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--early_stop', default=False, action="store_true")
    parser.add_argument('--threshold', default=1e-6, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--do_train', default=False, action="store_true")
    parser.add_argument('--do_test', default=False, action="store_true")

    d = vars(parser.parse_args())
    option = Option(d)
    option.tag = time.strftime("%y-%m-%d %H:%M:%S")
    if option.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(option.gpu_id)
    set_seed(option)
    dataset = Dataset(option.data_dir, option.batch_size, option.target_relation, option)
    # save_data(dataset.kg, dataset.entity2id, dataset.relation2id, option.exp_dir)
    if option.do_train:
        bl = option.save()
        print("Option saved.")
        start_time = time.time()
        main(dataset, option)
        print('Total time cost:', time.time() - start_time)

    if option.do_test:
        option.exp_dir = os.path.join(option.exps_dir, option.exp_name)
        ori_time = time.time()
        mrr_all, hit_1_all, hit_3_all, hit_10_all = test(dataset, option)
        print('Test Score: Mrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(mrr_all, hit_1_all, hit_3_all, hit_10_all))
        print('Current Time: {}, Total cost: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), time.time() - ori_time))
