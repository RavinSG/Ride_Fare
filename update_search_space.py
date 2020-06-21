import numpy as np
from collections import defaultdict


def load_from_history():
    file = open('logs/history.txt', 'r')
    history = defaultdict()
    count = 0

    for line in file:
        history[count] = eval(line)
        count += 1

    f1_scores = []

    for i, j in history.items():
        f1_scores.append(j['f1'])

    thresh = np.percentile(f1_scores, 95)
    indices = []

    for i, j in enumerate(f1_scores):
        if j > thresh:
            indices.append(i)

    return history, indices


def update_values():
    history, indices = load_from_history()

    learning_rate_values = []
    gamma_values = []
    max_depth_values = []
    sub_sample_values = []
    min_child_weight_values = []
    reg_lambda_values = []
    num_parallel_tree_values = []

    for i in indices:
        hist_val = history[i]
        learning_rate_values.append((hist_val['learning_rate']))
        gamma_values.append(hist_val['gamma'])
        max_depth_values.append(hist_val['max_depth'])
        min_child_weight_values.append(hist_val['min_child_weight'])
        sub_sample_values.append(hist_val['subsample'])
        reg_lambda_values.append(hist_val['reg_lambda'])
        num_parallel_tree_values.append(hist_val['num_parallel_tree'])

    gamma_max = max(gamma_values)
    gamma_min = min(gamma_values)

    depth_max = max(max_depth_values)
    depth_min = min(max_depth_values)

    child_weight_max = max(min_child_weight_values)
    child_weight_min = min(min_child_weight_values)

    sub_sample_max = max(sub_sample_values)
    sub_sample_min = min(sub_sample_values)

    lambda_max = max(reg_lambda_values)
    lambda_min = min(reg_lambda_values)

    trees_max = max(num_parallel_tree_values)
    trees_min = min(num_parallel_tree_values)

    lr_max = max(learning_rate_values)
    lr_min = min(learning_rate_values)

    updated_values = {
        'gamma': (gamma_min, gamma_max),
        'depth': (depth_min, depth_max),
        'child_weight': (child_weight_min, child_weight_max),
        'sub_sample': (sub_sample_min, sub_sample_max),
        'lambda': (lambda_min, lambda_max),
        'num_trees': (trees_min, trees_max),
        'lr': (lr_min, lr_max)
    }
    return updated_values
