import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
import time
from utils.utils import *
from utils.eval import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
import matplotlib.pyplot as plt
from thop import profile


def my_kl_loss(p, q):
    # p,q : B,H,L,L
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)  # B,L


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        loss = val_loss
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(val_loss, model, path)
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} because score {loss} '
                  f'> best_score {self.best_loss}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


def point_adjust(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    return pred


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               step=self.train_data_step,
                                               ratio=self.train_data_ratio, mode='train',
                                               dataset=self.dataset)
        self.train_nolap_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                                     step=self.train_data_step,
                                                     ratio=self.train_data_ratio, mode='train-nolap',
                                                     dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              step=self.train_data_step,
                                              ratio=self.train_data_ratio, mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              step=self.train_data_step,
                                              ratio=self.train_data_ratio, mode='thres',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

        if self.no_linear_attn:
            self.loss_fun = myLoss2
        else:
            self.loss_fun = myLossNew

        str1 = ''
        if self.no_point_adjustment:
            str1 = str1 + '_no_pa'
        if self.no_linear_attn:
            str1 = str1 + '_no_li'
        self.checkpoint_file = os.path.join(self.model_save_path, str(self.dataset) + '_checkpoint' + str1 + '.pth')

        log_file = os.path.join('logs', f'log-{self.dataset}.log')
        logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def build_model(self):
        linear_attn = not self.no_linear_attn
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3,
                                        linear_attn=linear_attn, mapping_fun=self.mapping_function)
        self.model2 = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3,
                                         linear_attn=linear_attn, mapping_fun=self.mapping_function)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()
            self.model2.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input_ = input_data.float().to(self.device)
            output, queries_list, keys_list = self.model(input_)
            len_list = len(queries_list)
            loss_attn = 0.0

            for u in range(len_list):
                loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside).mean()

            loss_attn = loss_attn / len_list

            rec_loss = self.criterion(output, input_)

            thisLoss1 = rec_loss
            thisLoss2 = rec_loss - self.k * loss_attn

            loss_1.append(thisLoss1.item())
            loss_2.append(thisLoss2.item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        info = "======================TRAIN MODE======================"
        print(info)
        logging.info(info)

        self.build_model()

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        # early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        params, flops = 0, 0
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            loss2_list = []
            epoch_start_time = time.time()
            self.model.train()

            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input_ = input_data.float().to(self.device)

                if epoch == 0 and i == 0:
                    input_profile = input_[[0]]
                    flops, params = profile(self.model2, inputs=(input_profile,))
                    flops = flops/1e9
                    params = params/1e6
                    epoch_start_time = time.time()

                output, queries_list, keys_list = self.model(input_)
                len_list = len(queries_list)

                # calculate Association discrepancy
                loss_attn = 0.0
                if not self.no_point_adjustment:
                    for u in range(len_list):
                        # b,l,h,d
                        loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside).mean()
                else:
                    loss_attn = 0

                loss_attn = loss_attn / len_list

                rec_loss = self.criterion(output, input_)

                loss1 = rec_loss
                loss2 = rec_loss - self.k * loss_attn  # loss_attn is used to distinguish normals and anomalies
                # test Equivalence
                loss3 = 2*rec_loss - self.k * loss_attn

                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())

                if (i + 1) % 50 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)

                    info = (
                        f'\t loss1: {loss1:.4f}, loss2: {loss2:.4f}; rec_loss: {rec_loss:.4f}, loss_attn: {loss_attn:.4f}'
                        f' speed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    print(info)
                    logging.info(info)

                    iter_count = 0
                    time_now = time.time()

                if not self.no_point_adjustment:
                    # using point adjustment
                    loss3.backward()
                    # loss1.backward()  # just for attention matrix plot
                else:
                    # no_point_adjustment
                    loss3.backward()

                self.optimizer.step()

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024.0)
            epoch_time = time.time() - epoch_start_time
            info = (f"Epoch: {epoch + 1} cost time: {epoch_time}, memory: {memory_used:.2f}GB, "
                    f"flops: {flops}GFLOPS, params: {params}M")
            print(info)
            logging.info(info)

            torch.save(self.model.state_dict(), self.checkpoint_file)
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        return epoch_time, flops, params, memory_used

    def test(self):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        self.model.eval()
        temperature = self.temperature
        softmax_span = self.softmax_span

        info = "======================TEST MODE======================"
        print(info)
        logging.info(info)

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        loss_list = []
        for i, (input_data, labels) in enumerate(self.train_nolap_loader):
            input = input_data.float().to(self.device)
            output, queries_list, keys_list = self.model(input)
            len_list = len(queries_list)
            # loss [B, L]  [256, 100]
            loss = torch.mean(criterion(input, output), dim=-1)
            loss_attn = 0.0
            for u in range(len_list):
                if u == 0:
                    # b,l
                    loss_attn = self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)  # * temperature
                else:
                    loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)  # * temperature

            # metric = torch.softmax(-loss_attn / len_list, dim=-1)
            loss_attn = loss_attn / len_list

            loss_attn = loss_attn.detach().cpu().numpy()
            attens_energy.append(loss_attn)
            loss_list.append(loss.detach().cpu().numpy())

        train_attn_array = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_loss_array = np.concatenate(loss_list, axis=0).reshape(-1)

        # aggregation
        if not self.no_point_adjustment:
            train_energy = softmax(-train_attn_array, temperature=temperature, window=softmax_span) * train_loss_array
        else:
            # no point adjustment
            train_energy = train_loss_array  # / np.maximum(train_attn_array, 1e-6)

        # (2) find the threshold
        attens_energy = []
        loss_list = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, queries_list, keys_list = self.model(input)
            len_list = len(queries_list)

            loss = torch.mean(criterion(input, output), dim=-1)

            loss_attn = 0.0
            for u in range(len_list):
                if u == 0:
                    # b,l
                    loss_attn = self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)
                else:
                    loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)

            # Metric
            loss_attn = loss_attn / len_list

            loss_attn = loss_attn.detach().cpu().numpy()
            attens_energy.append(loss_attn)
            loss_list.append(loss.detach().cpu().numpy())

        test_attn_array = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_loss_array = np.concatenate(loss_list, axis=0).reshape(-1)

        # aggregation
        if not self.no_point_adjustment:
            test_energy = softmax(-test_attn_array, temperature=temperature, window=softmax_span) * test_loss_array
        else:
            # no point adjustment
            # test_energy = test_loss_array / np.maximum(test_attn_array, 1e-6)
            # test_energy = test_loss_array * np.exp(-test_attn_array * self.attn_temp)
            test_energy = test_loss_array
            # test_energy = test_loss_array * -np.log(test_attn_array)
            # test_energy = np.maximum(softmax(-test_attn_array, temperature=temperature, window=softmax_span),
            #                          1 / np.maximum(test_attn_array, 1e-6)) * test_loss_array

        # probs
        if not self.no_gauss_dynamic and not self.no_point_adjustment:
            train_energy = get_probs_from_cri(train_energy)
            test_energy = get_probs_from_cri(test_energy)

        # smooth it out
        if self.no_point_adjustment:
            # no point adjustment; smooth it out
            train_energy = use_smooth(train_energy, kernel_length=self.kernel_length)
            test_energy = use_smooth(test_energy, kernel_length=self.kernel_length)

        # thres
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)

        info = f"Threshold : {thresh}"
        print(info)
        logging.info(info)

        combined_loss = np.concatenate([train_loss_array, test_loss_array], axis=0)  #
        thresh_rec_loss = np.percentile(combined_loss, 100 - self.anormly_ratio)

        info = f"Threshold for reconstruction loss: {thresh_rec_loss}"
        print(info)
        logging.info(info)

        # (3) evaluation on the test set
        test_labels = []
        test_data = []
        attens_energy = []

        loss_list = []

        eval_time = 0
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input_ = input_data.float().to(self.device)

            start_time = time.time()
            output, queries_list, keys_list = self.model(input_)
            eval_time += time.time() - start_time

            len_list = len(queries_list)

            loss = torch.mean(criterion(input_, output), dim=-1)

            loss_attn = []
            for u in range(len_list):
                if u == 0:
                    # b,l
                    loss_attn = self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)
                else:
                    loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)

            # Metric
            loss_attn = loss_attn / len_list

            loss_attn = loss_attn.detach().cpu().numpy()
            attens_energy.append(loss_attn)

            loss_list.append(loss.detach().cpu().numpy())

            # test_label
            test_labels.append(labels)
            # test data
            test_data.append(input_data)

        if self.record_state:
            # just for recording state
            return eval_time

        test_attn_array = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_att_loss2 = test_attn_array
        test_rec_loss = np.concatenate(loss_list, axis=0).reshape(-1)

        # time
        info = f"{self.dataset} Evaluation: cost time: {eval_time}"
        print(info)
        logging.info(info)

        # aggregation
        anomaly_score = None
        if not self.no_point_adjustment:
            test_att_loss2 = softmax(-test_attn_array, temperature=temperature, window=softmax_span)
            test_energy = test_att_loss2 * test_rec_loss
            anomaly_score = test_energy
        else:
            # no point adjustment
            # test_energy = test_loss_array / np.maximum(test_attn_array, 1e-6)
            # test_energy = test_loss_array * np.exp(-test_attn_array * self.attn_temp)
            test_energy = test_loss_array
            # test_energy = test_loss_array * -np.log(test_attn_array)
            # test_energy = np.maximum(softmax(-test_attn_array, temperature=temperature, window=softmax_span),
            #                          1 / np.maximum(test_attn_array, 1e-6)) * test_loss_array

        # use prob, not for non-point-adjustment
        if not self.no_gauss_dynamic and not self.no_point_adjustment:
            test_energy = get_probs_from_cri(test_energy)

        # smooth it out
        if self.no_point_adjustment:
            # no point adjustment; smooth it out
            test_energy = use_smooth(test_energy, kernel_length=self.kernel_length)

        # test labels
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        lmax, lmin = compute_longest_anomaly(test_labels)
        print(f'The anomaly span is [{lmin}-{lmax}]')
        print(f'anomaly ratio in test set: {sum(test_labels)} / {len(test_labels)} = '
              f'{sum(test_labels) / len(test_labels):.2%}')

        # b,l,d
        test_data = np.concatenate(test_data, axis=0).reshape(-1, test_data[0].shape[-1])



        # plot something
        if self.monte_carlo <= 1 and self.mode in ['test', 'monte_carlo']:

            myplot(test_labels, test_data, test_energy, test_rec_loss, test_attn_array, str(self.dataset), anomaly_score)
            print('here')
            # use test_attn_array instead of test_att_loss2

            # plot attn_matrix
            att_matrix = queries_list[-1].detach().cpu().numpy()
            if att_matrix.shape[-1] == att_matrix.shape[-2]:
                batch_idx = random.randint(0, att_matrix.shape[0] - 1)
                head_idx = random.randint(0, att_matrix.shape[1] - 1)
                att_matrix = att_matrix[batch_idx, head_idx, :, :]
                strflag = 'Vanilla_'
            else:
                # queries, keys: B, L, H, D
                batch_idx = random.randint(0, att_matrix.shape[0] - 1)
                head_idx = random.randint(0, att_matrix.shape[2] - 1)
                Q_mat = att_matrix[batch_idx, :, head_idx, :]
                K_mat = (keys_list[-1].detach().cpu().numpy())[batch_idx, :, head_idx, :]
                att_matrix = Q_mat @ K_mat.T
                att_matrix = att_matrix / att_matrix.sum(axis=1, keepdims=True)
                strflag = 'Linear_'
            plot_mat(att_matrix, strflag + str(self.dataset) + f"_batch_{batch_idx}_head_{head_idx}")

        gt = test_labels.astype(int)

        # get final pred
        pred = (test_energy > thresh).astype(int)
        # pred = np.random.rand(gt.shape[0]) > 0.8  # F:79
        if not self.no_point_adjustment:
            pred = point_adjust(pred, gt)

        info = f"pred: {pred.shape}"
        print(info)
        logging.info(info)
        info = f"gt: {gt.shape}"
        print(info)
        logging.info(info)

        ############    evaluate     ############
        print('------------Begin evaluating------------')
        # eval_dict = evaluate(pred, gt, validation_thresh=None)  # , false_pos_rates, true_pos_rates
        # if isinstance(eval_dict, tuple):
        #     eval_dict = eval_dict[0]
        # print('------------ eval Metrics -------------')
        # for k, v in eval_dict.items():
        #     info = f'{k}:\t{v}'
        #     print(info)
        #     logging.info(info)

        # detection adjustment: please see this issue for more information
        # https://github.com/thuml/Anomaly-Transformer/issues/14
        # pred = point_adjust(pred, gt)
        # gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)
        #
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')

        info = f"Accuracy : {accuracy:.2%}, Precision : {precision:.2%}, Recall : {recall:.2%}, F-score : {f_score:.2%}"
        print(info)
        logging.info(info)

        print('-------------- End ----------------')
        logging.info('-------------- End ----------------')

        # write into txt
        timestamp = time.strftime("%Y%m%d %H:%M:%S", time.localtime())
        os.makedirs('output', exist_ok=True)
        with open(os.path.join('output', str(self.dataset) + '-result.txt'), 'a') as f:
            f.write(timestamp + '\n')
            f.write(f"\tAccuracy : {accuracy:.2%}, Precision : {precision:.2%}, Recall : {recall:.2%}, "
                    f"F-score : {f_score:.2%}" + '\n')
            # for k, v in eval_dict.items():
            #     f.write(f'\t{k}:\t{v}' + '\n')
            f.write('\n')

        return accuracy, precision, recall, f_score
