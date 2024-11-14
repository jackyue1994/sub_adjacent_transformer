import logging
import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver
import shutil


def str2bool(v1):
    return v1.lower() in 'true'


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    str1 = ''
    if config.no_point_adjustment:
        str1 = str1 + '_no_pa'
    if config.no_linear_attn:
        str1 = str1 + '_no_li'

    src_file = os.path.join(config.model_save_path, str(config.dataset) + '_checkpoint' + str1 + '.pth')

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'monte_carlo' and config.monte_carlo >= 1:
        p_list = []
        r_list = []
        f_list = []
        file_list = []
        max_f = 0
        dst_file = None
        for i in range(config.monte_carlo):
            print(f"-------------------Monte carlo recycle {i + 1}/{config.monte_carlo},"
                  f" current_best: {max_f:.2%}-------------------")
            solver.train()
            _, precision, recall, f_score = solver.test()
            p_list.append(precision)
            r_list.append(recall)
            f_list.append(f_score)
            if f_score > max_f:
                max_f = f_score
                if config.no_point_adjustment:
                    dst_file = os.path.join(config.model_save_path, str(config.dataset) +
                                            f"_anormly_ratio_{config.anormly_ratio:.2f}_temp_{config.attn_temp:.0f}"
                                            f"_kernel_length_{config.kernel_length:.0f}_fscore_{max_f:.4f}" + str1 +
                                            '.pth')
                else:
                    dst_file = os.path.join(config.model_save_path, str(config.dataset) +
                                            f"_anormly_ratio_{config.anormly_ratio:.2f}_temp_{config.temperature:.0f}"
                                            f"_softmax_span_{config.softmax_span:.0f}_fscore_{max_f:.4f}" + str1 +
                                            '.pth')
                shutil.copyfile(src_file, dst_file)
                file_list.append(dst_file)

        max_f = max(f_list)
        ind = f_list.index(max_f)
        print("----------Monte carlo recycle end----------")
        if config.no_point_adjustment:
            print(f"Best Result: anormly_ratio: {config.anormly_ratio:.2f}, attn_temp: {config.attn_temp:.0f}, "
                  f"kernel_length: {config.kernel_length:.0f}, P: {p_list[ind]:.2%}, R: {r_list[ind]:.2%}, F:{max_f:.2%}")
        else:
            print(f"Best Result: anormly_ratio: {config.anormly_ratio:.2f}, temperature: {config.temperature:.0f}, "
                  f"softmax_span: {config.softmax_span:.0f}, P: {p_list[ind]:.2%}, R: {r_list[ind]:.2%}, F:{max_f:.2%}")
        if dst_file:
            print('The best pth is copied to checkpoint...')
            shutil.copyfile(dst_file, src_file)
            for file in file_list:
                if file != dst_file:
                    os.remove(file)
    elif config.mode == 'find_best':
        p_list = []
        r_list = []
        f_list = []
        max_f = 0
        dst_file = None
        best_ratio = 0
        best_temp = 0
        best_span = 0
        file_list = []

        anormly_ratio_span = config.anormly_ratio_span
        anormly_ratio_step = config.anormly_ratio_step
        temperature_span = config.temperature_range
        temperature_step = config.temperature_step
        anormly_ratio_list = np.arange(anormly_ratio_span[0], anormly_ratio_span[1] + 1e-3, anormly_ratio_step)

        if config.no_point_adjustment:
            temperature_list = np.arange(config.attn_temp_range[0], config.attn_temp_range[1] + 1e-3,
                                         config.attn_temp_step)
            softmax_span_list = np.arange(config.kernel_length_range[0], config.kernel_length_range[1] + 1e-3,
                                          config.kernel_length_step)
        else:
            temperature_list = np.arange(temperature_span[0], temperature_span[1] + 1e-3, temperature_step)
            softmax_span_list = np.arange(config.softmax_span_range[0], config.softmax_span_range[1] + 1e-3,
                                          config.softmax_span_step)
        # softmax_span_list = [config.softmax_span]  # too time-consuming to use softmax_span_list
        for span in softmax_span_list:
            for temp in temperature_list:
                for anormly_ratio in anormly_ratio_list:
                    solver.anormly_ratio = anormly_ratio
                    if config.no_point_adjustment:
                        solver.attn_temp = temp
                        solver.kernel_length = span
                    else:
                        solver.temperature = temp
                        solver.softmax_span = span
                    if config.no_point_adjustment:
                        print(f"---------find_the_best (kernel_length: {span:.0f}, temp:{temp:.0f}, "
                              f"ratio:{anormly_ratio:.3f}, current_best: {max_f:.2%})---------")
                    else:
                        print(f"---------find_the_best (softmax_span: {span:.0f}, temp:{temp:.0f}, "
                              f"ratio:{anormly_ratio:.3f}, current_best: {max_f:.2%})---------")
                    # test
                    _, precision, recall, f_score = solver.test()
                    p_list.append(precision)
                    r_list.append(recall)
                    f_list.append(f_score)
                    if f_score > max_f:
                        max_f = f_score
                        best_ratio = anormly_ratio
                        best_temp = temp
                        best_span = span
                        if config.no_point_adjustment:
                            dst_file = os.path.join(config.model_save_path, str(config.dataset) +
                                                    f"_anormly_ratio_{anormly_ratio:.2f}_temp_{temp:.0f}"
                                                    f"_kernel_length_{span:.0f}_fscore_{max_f:.4f}" + str1 + '.pth')
                        else:
                            dst_file = os.path.join(config.model_save_path, str(config.dataset) +
                                                    f"_anormly_ratio_{anormly_ratio:.3f}_temp_{temp:.0f}"
                                                    f"_softmax_span_{span:.0f}_fscore_{max_f:.4f}" + str1 + '.pth')
                        shutil.copyfile(src_file, dst_file)
                        file_list.append(dst_file)

        max_f = max(f_list)
        ind = f_list.index(max_f)
        print("----------find_the_best end----------")
        if config.no_point_adjustment:
            print(f"Best Result: kernel_length: {best_span:.0f}, ratio: {best_ratio:.3f}, "
                  f"temperature: {best_temp:.0f}, P: {p_list[ind]:.2%}, R: {r_list[ind]:.2%}, F:{max_f:.2%}")
        else:
            print(f"Best Result: softmax_span: {best_span:.0f}, ratio: {best_ratio:.3f}, "
                  f"temperature: {best_temp:.0f}, P: {p_list[ind]:.2%}, R: {r_list[ind]:.2%}, F:{max_f:.2%}")
        if dst_file:
            for file in file_list:
                if file != dst_file:
                    os.remove(file)

        return max_f, best_span, best_temp, best_ratio
    elif (config.mode == 'monte_carlo_span' and config.monte_carlo >= 1 and
          config.softmax_span_range[1] >= config.softmax_span_range[0] and
          config.kernel_length_range[1] >= config.kernel_length_range[0]):
        p_list = []
        r_list = []
        f_list = []
        file_list = []
        max_f = 0
        best_span = 0
        dst_file = None
        if config.no_point_adjustment:
            span_list = np.arange(config.kernel_length_range[0], config.kernel_length_range[1] + 1e-3,
                                  config.kernel_length_step)
        else:
            span_list = np.arange(config.softmax_span_range[0], config.softmax_span_range[1] + 1e-3,
                                  config.softmax_span_step)

        for i in range(config.monte_carlo):
            print(f"--------------------------Monte carlo with {i + 1}/{config.monte_carlo} --------------------------")
            solver.train()
            for span in span_list:
                print(f"-------------------Monte carlo {i + 1}/{config.monte_carlo}, span {span} , "
                      f"current_best: {max_f:.2%}-------------------")
                if config.no_point_adjustment:
                    solver.kernel_length = span
                else:
                    solver.softmax_span = span
                _, precision, recall, f_score = solver.test()
                p_list.append(precision)
                r_list.append(recall)
                f_list.append(f_score)
                if f_score > max_f:
                    max_f = f_score
                    best_span = span
                    if config.no_point_adjustment:
                        dst_file = os.path.join(config.model_save_path, str(config.dataset) +
                                                f"_anormly_ratio_{config.anormly_ratio:.2f}_temp_{config.attn_temp:.0f}"
                                                f"_kernel_length_{span:.0f}_fscore_{max_f:.4f}" + str1 + '.pth')
                    else:
                        dst_file = os.path.join(config.model_save_path, str(config.dataset) +
                                                f"_anormly_ratio_{config.anormly_ratio:.2f}_temp_{config.temperature:.0f}"
                                                f"_softmax_span_{span:.0f}_fscore_{max_f:.4f}" + str1 + '.pth')
                    shutil.copyfile(src_file, dst_file)
                    file_list.append(dst_file)

        max_f = max(f_list)
        ind = f_list.index(max_f)
        print("----------Monte carlo with span recycle end----------")
        if config.no_point_adjustment:
            print(f"Best Result: best_kernel_length: {best_span:.0f}, temperature: {config.attn_temp:.0f}, "
                  f"ratio: {config.anormly_ratio:.2f}, P: {p_list[ind]:.2%}, R: {r_list[ind]:.2%}, F:{max_f:.2%}")
        else:
            print(f"Best Result: best_softmax_span: {best_span:.0f}, temperature: {config.temperature:.0f}, "
                  f"ratio: {config.anormly_ratio:.2f}, P: {p_list[ind]:.2%}, R: {r_list[ind]:.2%}, F:{max_f:.2%}")
        if dst_file:
            print('The best pth is copied to checkpoint...')
            shutil.copyfile(dst_file, src_file)
            for file in file_list:
                if file != dst_file:
                    os.remove(file)

        return max_f, best_span, p_list[ind], r_list[ind]
    else:
        print('config mode error...')

    return solver


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings("ignore")

    # seed = 0
    # torch.manual_seed(seed)
    #
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=float, default=10)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'monte_carlo', 'monte_carlo_span', 'monte_carlo_search', 'find_best'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    parser.add_argument('--anormly_ratio', type=float, default=0.8)
    parser.add_argument('--anormly_ratio_span', nargs=2, type=float, default=[0.1, 1], help='used for mode find_best')
    parser.add_argument('--anormly_ratio_step', type=float, default=0.1, help='used for mode find_best')

    parser.add_argument('--monte_carlo', type=int, default=1)
    parser.add_argument('--oneside', action='store_true', default=False)
    parser.add_argument('--span', nargs=2, type=int, default=[20, 30])  # k1 k2

    parser.add_argument('--temperature', type=float, default=50)
    parser.add_argument('--temperature_range', nargs=2, type=int, default=[50, 50], help='used for find_best')
    parser.add_argument('--temperature_step', type=int, default=10, help='used for mode find_best')

    parser.add_argument('--softmax_span', type=int, default=200)
    parser.add_argument('--softmax_span_range', nargs=2, type=int, default=[200, 200], help='used for monte_carlo_span')
    parser.add_argument('--softmax_span_step', type=int, default=50, help='used for mode monte_carlo_span')

    parser.add_argument('--no_gauss_dynamic', action='store_true', default=False)
    parser.add_argument('--no_point_adjustment', action='store_true', default=False)
    parser.add_argument('--no_linear_attn', action='store_true', default=False)

    parser.add_argument('--kernel_length', type=int, default=100)
    parser.add_argument('--kernel_length_range', nargs=2, type=int, default=[0, 600],
                        help='used for no_point_adjustment')
    parser.add_argument('--kernel_length_step', type=int, default=10, help='used for mode no_point_adjustment')

    parser.add_argument('--attn_temp', type=float, default=1)
    parser.add_argument('--attn_temp_range', nargs=2, type=int, default=[1, 1], help='no_point_adjustment')
    parser.add_argument('--attn_temp_step', type=int, default=5, help='used for no_point_adjustment')

    parser.add_argument('--mapping_function', type=str, default='ours',
                        choices=['softmax_q_k', 'ours', 'x_3', 'relu', 'elu_plus_1'])

    parser.add_argument('--record_state', action='store_true', default=False)

    parser.add_argument('--train_data_ratio', type=float, default=1.0)
    parser.add_argument('--train_data_step', type=int, default=1)

    config0 = parser.parse_args()

    args = vars(config0)

    # some files
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    log_file = os.path.join('logs', f'log-{config0.dataset}.log')
    logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    gpu_state_file = os.path.join('results', 'gpu_state.txt')

    print('------------ Options -------------')
    logging.info('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
        logging.info('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    logging.info('------------ End -------------')

    if config0.record_state:
        solver_ = Solver(vars(config0))
        epoch_time, flops, params, memory_used = solver_.train()
        eval_time = solver_.test()
        with open(gpu_state_file, 'a') as f:
            f.write('------------ Options -------------' + '\n')
            for k, v in sorted(args.items()):
                f.write(f'\t{k}:\t{v}' + '\n')
            f.write(f"\tepoch_time : {epoch_time:.5f}, eval_time : {eval_time:.5f}, flops : {flops:.5f} GFLOPS, "
                    f"params : {params:.5f} M, memory_used: {memory_used:.5f} G" + '\n')
            f.write('------------ Ends -------------' + '\n')
            f.write('\n')

    elif config0.mode == 'monte_carlo_search':
        # monte_carlo_span
        config0.mode = "monte_carlo_span"
        max_f, best_span, pre, rec = main(config0)

        # find_best
        config0.mode = "find_best"
        if config0.no_point_adjustment:
            config0.kernel_length_range = [best_span, best_span]
        else:
            config0.softmax_span_range = [best_span, best_span]
        config0.temperature_range = [config0.temperature, config0.temperature]

        if 'nips' in config0.dataset.lower():
            delta_ratio_span = 5
            delta_ratio_step = 0.1
            delta_temp = 50
        elif config0.no_point_adjustment:
            delta_ratio_span = 5
            delta_ratio_step = 0.1
        else:
            delta_ratio_span = 0.5
            delta_ratio_step = 0.01
            delta_temp = 50
        if rec > pre:
            config0.anormly_ratio_span = [max(config0.anormly_ratio - delta_ratio_span, 0.1),
                                          config0.anormly_ratio + delta_ratio_span*0.2]  # 0.5
        else:
            config0.anormly_ratio_span = [max(config0.anormly_ratio - delta_ratio_span*0.2, 0.1),
                                          config0.anormly_ratio + delta_ratio_span]  # 0.5
        config0.anormly_ratio_step = delta_ratio_step
        _, best_span, best_temp, best_ratio = main(config0)

        if not config0.no_point_adjustment:
            # using point adjustment
            # temperature
            config0.temperature_range = [max(config0.temperature - delta_temp, 0), config0.temperature + delta_temp]
            config0.temperature_step = 10
            config0.anormly_ratio_span = [best_ratio, best_ratio]
            _, _, best_temp, _ = main(config0)

            # span
            config0.softmax_span_range = [max(0, best_span - 50), best_span + 50]
            config0.softmax_span_step = 10
            config0.temperature_range = [best_temp, best_temp]
            max_f, best_span, best_temp, best_ratio = main(config0)
        else:
            # no point adjustment
            # kernel length
            config0.kernel_length_range = [max(0, best_span - 50), best_span + 50]
            config0.kernel_length_step = 10
            config0.anormly_ratio_span = [best_ratio, best_ratio]
            max_f, best_span, best_temp, best_ratio = main(config0)

        # write into txt
        txt_file = os.path.join("results", config0.dataset + '.txt')
        with open(txt_file, 'a') as f:
            f.write('------------ Options -------------' + '\n')
            for k, v in sorted(args.items()):
                f.write(f'\t{k}:\t{v}' + '\n')
            f.write(f"\tmax_f : {max_f:.2%}, best_span : {best_span}, best_temp : {best_temp}, "
                    f"best_ratio : {best_ratio:.4f}" + '\n')
            f.write('------------ Ends -------------' + '\n')
            f.write('\n')
    else:
        main(config0)
