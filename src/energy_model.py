import argparse
import csv
import datetime
import pickle
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score, \
    classification_report
import os.path


def print_matrix(text, matrix):
    print(text)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print("{:.2f} ".format(matrix[i][j]), end="")
        print()


# 1. Determine transition probabilities between stages
def find_transition_prob(data):
    unique, counts = np.unique(data, return_counts=True)
    print("Unique values {} of labels {}".format(counts, unique))
    data_prev = data[:-1]
    data_next = data[1:]
    n_labels = len(unique)
    trans_prob = np.zeros((n_labels, n_labels))
    for label in unique:
        transitions = data_next[np.where(data_prev == label)]
        trans_count = np.bincount(transitions)
        trans_prob[label] = np.append(trans_count, np.zeros(n_labels - len(trans_count)))
        trans_prob[label] /= np.sum(trans_prob[label])

    return trans_prob

# 2. Finds the energy of the state of each epoch in the hypnogram.
# The energy depends on the probability of each state, as defined in the paper.
def find_energy_state_vec(alpha, conf_real_pred, e_pred, e_trans, pred, pred_opt, trans_prob, pred_prob):

    error_t1 = np.zeros(len(pred))
    error_t2 = np.zeros(len(pred))
    error_pc = np.zeros(len(pred))
    error_pr = np.zeros(len(pred))

    for i in range(len(pred)):
        state = pred_opt[i]
        # Probability for each state transition from state series in the training set
        if i > 0:
            p_trans = trans_prob[pred_opt[i - 1]][state]
            error_t1[i] = (1 + e_trans) / (p_trans + e_trans) - 1
        if i < len(pred) - 1:
            p_trans = trans_prob[state][pred_opt[i + 1]]
            error_t2[i] = (1 + e_trans) / (p_trans + e_trans) - 1
        # Probability for each state from the confusion matrices from the validation set
        p_pred = conf_real_pred[state][pred[i]]
        error_pc[i] = (1 + e_pred) / (p_pred + e_pred) - 1
        # Probability for each state from the NN predictions for the state
        p_out = pred_prob[i][state]
        error_pr[i] = (1 + e_pred) / (p_out + e_pred) - 1

    energy_vec = alpha[0] * (error_t1 + error_t2) + alpha[1] * (error_pc + error_pr)
    return energy_vec

# 3. Finds the changes in energy of the state of a single epoch
def find_delta_energy(curr_energy, alpha, conf_real_pred, e_pred, e_trans, i,
                      pred, pred_opt, state, trans_prob, pred_prob):
    # Probability for each state transition from state series in the training set
    error_t, error_t2 = 0, 0
    if i > 0:
        p_trans = trans_prob[pred_opt[i - 1]][state]
        error_t = (1 + e_trans) / (p_trans + e_trans) - 1
    if i < len(pred) - 1:
        p_trans = trans_prob[state][pred_opt[i + 1]]
        error_t2 = (1 + e_trans) / (p_trans + e_trans) - 1
    # Probability for each state from the confusion matrices from the validation set
    p_pred = conf_real_pred[state][pred[i]]
    error_p1 = (1 + e_pred) / (p_pred + e_pred) - 1
    # Probability for each state from the NN predictions for the state
    p_out = pred_prob[i][state]
    error_p2 = (1 + e_pred) / (p_out + e_pred) - 1

    delta = np.array([0, 0, 0])
    delta[1] = alpha[0] * (error_t + error_t2) + alpha[1] * (error_p1 + error_p2) - curr_energy

    # Evaluate the energy change of previous and next positions
    if i > 0:
        p_transP = trans_prob[pred_opt[i - 1]][pred_opt[i]]
        p_transN = trans_prob[pred_opt[i - 1]][state]
        delta[0] = alpha[0]*((1 + e_trans) / (p_transN + e_trans) - (1 + e_trans) / (p_transP + e_trans))
    if i < len(pred) - 1:
        p_transP = trans_prob[pred_opt[i]][pred_opt[i+1]]
        p_transN = trans_prob[state][pred_opt[i+1]]
        delta[2] = alpha[0]*((1 + e_trans) / (p_transN + e_trans) - (1 + e_trans) / (p_transP + e_trans))

    return delta


# 4. Optimize sleep series using the energy function
# trans_prob: transition probability [prev_state][next_state]
# conf_real_pred: confusion matrix [real][pred]
def optimize_energy(trans_prob, conf_real_pred, pred, pred_prob, target, betas,
                    e_trans=0.1, e_pred=0.1, alpha=None, n_steps=100):
    if alpha is None:
        alpha = [1.0, 1.0]
    n_states = len(trans_prob)
    pred_opt = pred.copy()

    # Finds the error for each position
    energy_list = find_energy_state_vec(alpha, conf_real_pred, e_pred, e_trans, pred, pred_opt, trans_prob, pred_prob)

    # Changes one position on each step
    for step in range(n_steps):
        if step > 0 and step % 1000 == 0 and len(target) == len(pred_opt):
            print("Step {:5} Accuracy: {:.2f}%".format(step,
                                                       100 * len(np.equal(target, pred_opt).nonzero()[0]) / len(
                                                           target)))

        # Select position to update. Selection probability is proportional to position energy
        pos = np.random.choice(len(energy_list), 1, p=softmax(betas[step] * energy_list))[0]

        # Select the state of the selected position. Next states with less energy are more likely to be selected
        delta_energy = []  # np.zeros(n_states)
        for state in range(n_states):
            delta_energy.append(find_delta_energy(energy_list[pos], alpha, conf_real_pred, e_pred, e_trans, pos, pred,
                                                  pred_opt, state, trans_prob, pred_prob))
        delta_sum = np.sum(delta_energy, axis=1)
        # best_state = np.argmin(delta_sum)
        best_state = np.random.choice(len(delta_sum), 1, p=softmax(-betas[step] * delta_sum))[0]

        # Update the state of the selected position and its neighbors.
        pred_opt[pos] = best_state
        energy_list[pos] += delta_energy[best_state][1]
        if pos > 0:
            energy_list[pos-1] += delta_energy[best_state][0]
        if pos < len(energy_list) - 1:
            energy_list[pos+1] += delta_energy[best_state][2]

    return pred_opt

# 5. Optimize sleep series for multiple subjects in a dataset
def perform_optimizations_dataset(opt_params, part, model, dataset, writefile):

    pickle_input = "../data/input_" + dataset + "_" + model + "_part_" + str(part) + ".pkl"
    file = open(pickle_input, 'rb')
    data_per_subj = pickle.load(file)
    file.close()

    # Concatenate validation data from all subjects
    y_true_valid = []
    y_pred_valid = []
    y_prob_valid = []
    for index, row in data_per_subj[data_per_subj['train_test'] == 'valid'].iterrows():
        y_true_valid.append(row.y_true)
        y_pred_valid.append(row.y_pred)
        y_prob_valid.append(row.y_prob)
    y_true_valid = np.hstack(y_true_valid)
    y_pred_valid = np.hstack(y_pred_valid)
    y_prob_valid = np.vstack(y_prob_valid)

    # Determine Transition Probability matrix
    y_true_train = []
    for index, row in data_per_subj[data_per_subj['train_test'] == 'train'].iterrows():
        y_true_train.append(row.y_true)
    y_true_train = np.hstack(y_true_train)
    trans_prob_train = find_transition_prob(y_true_train)

    # output files
    csv_file = '../output/sleep-results-' + dataset + '.csv'
    npz_file = "../output/" + "energy-" + model + "-" + str(part) + "-" + dataset + "-opt.npz"

    curr_time = '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
    targets = ['0', '1', '2', '3', '4']

    # Creates and normalizes confusion matrix from validation data
    conf_mat_valid = confusion_matrix(y_true_valid, y_pred_valid)
    conf_mat_valid = conf_mat_valid.astype('float') / conf_mat_valid.sum(axis=0)[:, np.newaxis]

    # Print initial metrics
    print_matrix('Confusion Matrix (Validation Set)', conf_mat_valid)
    acc_valid = accuracy_score(y_true_valid, y_pred_valid) * 100
    bal_valid = balanced_accuracy_score(y_true_valid, y_pred_valid) * 100
    print("Accuracy: {:.2f}% (Validation)".format(acc_valid))
    print("Balanced: {:.2f}% (Validation)".format(bal_valid))
    report = classification_report(y_true_valid, y_pred_valid, target_names=targets, output_dict=True)
    print_matrix('Transition Matrix (Training Set)', trans_prob_train)

    # Optimization model parameters (alpha, beta, and epsilons). Check paper for details.
    a_max_list, b_max, ep_max, et_max= [[0.5, 0.5]], 1.0, 0.1, 0.1
    if opt_params:
        a_max_list = [[0,1],[0.25,0.75],[0.5,0.5],[0.75,0.25],[1,0]]

    # Number of otimization steps and schedule for the beta (1/temperature) value
    n_steps = 5 * 1200
    sched_steps = [1, 2, 4, 8]
    beta_sched = []
    for b in sched_steps:
        beta_sched.append(np.ones(n_steps)*b)
    beta_sched = np.array(beta_sched).flatten()

    if 'recording' not in data_per_subj.columns:
        data_per_subj['recording'] = np.zeros(len(data_per_subj), dtype='int')

    y_test_list = {'pred': [], 'pred-opt': [], 'true': []}

    print('\n\n===== Evaluating Optimizations =====')
    metrics = {}
    for index, row in data_per_subj[data_per_subj['train_test'] == 'test'].iterrows():
        subj, rec = row['subjects'], row['recording']
        for a_max in a_max_list:
            print(subj, rec)
            subj_data = data_per_subj.loc[data_per_subj['subjects'] == subj].loc[data_per_subj['recording'] == rec]
            if subj_data.size == 0:
                continue
            y_pred_test = subj_data.y_pred.values[0]
            y_prob_test = subj_data.y_prob.values[0]
            y_true_test = subj_data.y_true.values[0]
            # Apply energy-optimization for a single subject
            y_pred_opt_test = optimize_energy(trans_prob_train, conf_mat_valid, y_pred_test, y_prob_test,
                                              [], b_max * beta_sched, et_max, ep_max, a_max, n_steps)

            metrics = evaluate_metrics(metrics, targets, y_pred_opt_test, y_pred_test, y_true_test, subj, rec, a_max)
            if writefile:
                write_results_file(opt_params, part, model, csv_file, metrics, curr_time,
                                   a_max, ep_max, et_max, targets, y_pred_opt_test, y_pred_test, y_true_test)

            y_test_list['pred-opt'].append(y_pred_opt_test)
            y_test_list['pred'].append(y_pred_test)
            y_test_list['true'].append(y_true_test)

    # Write results to file
    if writefile:
        np.savez(npz_file, conf_mat_valid=conf_mat_valid, conf_mat_test=metrics['cm_test'],
                 conf_mat_opt=metrics['cm_opt'], trans_prob_train=trans_prob_train,
                 subj=metrics['subj'], rec=metrics['rec'], alpha=metrics['alpha'],
                 y_pred_opt_test_list=y_test_list['pred-opt'], y_pred_test_list=y_test_list['pred'],
                 y_true_test=y_test_list['true'])

    print("\n================================================================")
    print("Finished - model: " + model + " cross-validation: " + str(part) + " dataset: " + dataset)
    print("mean_acc =", np.mean(metrics['acc']), "mean_acc_opt =", np.mean(metrics['acc_opt']))
    print("std_acc  =", np.std(metrics['acc']),  "std_acc_opt  =", np.std(metrics['acc_opt']))
    print("mean_bal =", np.mean(metrics['bal']), "mean_bal_opt =", np.mean(metrics['bal_opt']))
    print("std_bal  =", np.std(metrics['bal']),  "std_bal_opt  =", np.std(metrics['bal_opt']))
    print("================================================================ \n")


def write_results_file(optparams, part, model, csv_file, metr, curtime, a_max, ep_max, et_max, targets,
                       y_pred_opt_test, y_pred_test, y_true_test, pos=-1):

    if not os.path.exists(csv_file):
        row = ['type', 'part', 'model', 'optparams', 'et_max', 'ep_max', 'a_max',
               'acc', 'acc_opt', 'bal', 'bal_opt', 'coh', 'coh_opt', 'timestamp']
        for t in targets:
            row += ['acc-' + t, 'acc-opt-' + t, 'prec-' + t, 'prec-opt-' + t, 'rec-' + t, 'rec-opt-' + t,
                    'f1-' + t, 'f1-opt-' + t, 'sup-' + t, 'sup-opt-' + t]
        row += ['subject', 'record']
        with open(csv_file, 'a+', newline='') as write_obj:
            csv.writer(write_obj).writerow(row)
            write_obj.flush()

    report = classification_report(y_true_test, y_pred_test,
                                   target_names=targets, output_dict=True, labels=list(map(int, targets)))
    report_opt = classification_report(y_true_test, y_pred_opt_test,
                                       target_names=targets, output_dict=True, labels=list(map(int, targets)))
    results = ['energy', part, model, optparams, et_max, ep_max, a_max,
               metr['acc'][pos], metr['acc_opt'][pos], metr['bal'][pos], metr['bal_opt'][pos], metr['coh'][pos],
               metr['coh_opt'][pos], curtime]
    for t in targets:
        acc_t = metr['acc_class_test'][pos][int(t)]
        acc_opt_t = metr['acc_class_opt'][pos][int(t)]
        if np.isnan(acc_t):
            acc_t = 0
        if np.isnan(acc_opt_t):
            acc_opt_t = 0
        results += [acc_t, acc_opt_t,
                    report[t]['precision'], report_opt[t]['precision'],
                    report[t]['recall'], report_opt[t]['recall'],
                    report[t]['f1-score'], report_opt[t]['f1-score'],
                    report[t]['support'], report_opt[t]['support']]
    results += [metr['subj'][pos], metr['rec'][pos]]
    with open(csv_file, 'a+', newline='') as write_obj:
        csv.writer(write_obj).writerow(results)
        write_obj.flush()


# acc, acc_class_opt, acc_class_test, acc_opt, bal, bal_opt, cm_opt, cm_test, coh, coh_opt
def evaluate_metrics(metrics, targets, y_pred_opt_test, y_pred_test, y_true_test, subj=-1, rec=-1, alpha=None,
                     print_res=True):

    if alpha is None:
        alpha = [1.0, 1.0]
    if len(metrics) == 0:
        metrics = {'acc': [], 'acc_opt': [], 'bal': [], 'bal_opt': [], 'coh': [], 'coh_opt': [],
                   'cm_test': [], 'cm_opt': [], 'acc_class_test': [], 'acc_class_opt': [],
                   'y_true_counts': [], 'y_pred_counts': [], 'y_opt_counts': [], 'subj': [], 'rec': [], 'alpha': []}

    labels = list(map(int, targets))
    metrics['acc'].append(accuracy_score(y_true_test, y_pred_test))
    metrics['acc_opt'].append(accuracy_score(y_true_test, y_pred_opt_test))
    metrics['bal'].append(balanced_accuracy_score(y_true_test, y_pred_test))
    metrics['bal_opt'].append(balanced_accuracy_score(y_true_test, y_pred_opt_test))
    metrics['coh'].append(cohen_kappa_score(y_true_test, y_pred_test))
    metrics['coh_opt'].append(cohen_kappa_score(y_true_test, y_pred_opt_test))
    metrics['cm_test'].append(confusion_matrix(y_true_test, y_pred_test, normalize='true', labels=labels))
    metrics['cm_opt'].append(confusion_matrix(y_true_test, y_pred_opt_test, normalize='true', labels=labels))
    metrics['subj'].append(subj)
    metrics['rec'].append(rec)
    metrics['alpha'].append(alpha)

    metrics['acc_class_test'].append(metrics['cm_test'][-1].diagonal() / metrics['cm_test'][-1].sum(axis=1))
    metrics['acc_class_opt'].append(metrics['cm_opt'][-1].diagonal() / metrics['cm_opt'][-1].sum(axis=1))
    metrics['y_true_counts'].append(get_targets_counts(targets, y_true_test))
    metrics['y_pred_counts'].append(get_targets_counts(targets, y_pred_test))
    metrics['y_opt_counts'].append(get_targets_counts(targets, y_pred_opt_test))

    if print_res:
        print("Accuracy: {:.2f}% (Test Set)".format(metrics['acc'][-1] * 100))
        print("Accuracy: {:.2f}% (Optimized Test Set)".format(metrics['acc_opt'][-1] * 100))
        print("Balanced Accuracy: {:.2f}% (Test Set)".format(metrics['bal'][-1] * 100))
        print("Balanced Accuracy: {:.2f}% (Optimized Test Set)".format(metrics['bal_opt'][-1] * 100))
        print("Cohen Kappa: {:.2f}% (Test Set)".format(metrics['coh'][-1] * 100))
        print("Cohen Kappa: {:.2f}% (Optimized Test Set)".format(metrics['coh_opt'][-1] * 100))
        print("Counts y_true: ")
        print(metrics['y_true_counts'][-1] / sum(metrics['y_true_counts'][-1]))
        print("Counts y_pred_test: ")
        print(metrics['y_pred_counts'][-1] / sum(metrics['y_pred_counts'][-1]))
        print("Counts y_pred_opt_test: ")
        print(metrics['y_opt_counts'][-1] / sum(metrics['y_opt_counts'][-1]))
        # print(classification_report(y_true_test, y_pred_test, target_names=targets, labels=labels))
        # print(classification_report(y_true_test, y_pred_opt_test, target_names=targets, labels=labels))
        # print_matrix('Confusion Matrix (Test Set)', metr['cm_test'][-1])
        # print_matrix('Confusion Matrix (Optimized Set)', metr['cm_opt'][-1])

    return metrics


def get_targets_counts(targets, y_true_test):
    values, counts = np.unique(y_true_test, return_counts=True)
    if len(values) < len(targets):
        counts_new = np.zeros(len(targets))
        for t, c in zip(values, counts):
            counts_new[t] = c
        counts = counts_new
    return counts


def main(cmd_args):
    opt_params = cmd_args.optparams
    writefile = cmd_args.writefile

    if cmd_args.part == -1:
        part = [0, 1, 2, 3, 4]
    else:
        part = [cmd_args.part]

    if cmd_args.model == 'all':
        model = ['stager', 'usleep']
    else:
        model = [cmd_args.model]

    if cmd_args.dataset == 'all':
        dataset = ['edf', 'dreamer']
    else:
        dataset = [cmd_args.dataset]

    for d in dataset:
        for m in model:
            for p in part:
                perform_optimizations_dataset(opt_params, p, m, d, writefile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Energy model')

    parser.add_argument('--optparams', type=int, default=0,
                        help='Set 1 to optimize the energy model parameters and 0 otherwise.')
    parser.add_argument('--part', type=int, default=0,
                        help='Define the CV slice, between 0 and 5. Use -1 for all parts.')
    parser.add_argument('--model', type=str, default='usleep',
                        help='Name of the neural network model.', choices=['stager', 'usleep', 'all'])
    parser.add_argument('--writefile', type=int, default=1,
                        help='Set 1 to write results to file and 0 otherwise.')
    parser.add_argument('--dataset', type=str, default='edf',
                        help='Name of the dataset to optimize.', choices=['edf', 'dreamer', 'all'])

    args = parser.parse_args()
    main(args)
