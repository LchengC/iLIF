import torch
import random
import numpy as np

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_f1(gold, predicted, logger):
    c_predict = 0
    c_correct = 0
    c_gold = 0

    for g, p in zip(gold, predicted):
        if g != 0:
            c_gold += 1
        if p != 0:
            c_predict += 1
        if g != 0 and p != 0:
            c_correct += 1

    p = c_correct / (c_predict + 1e-100) if c_predict != 0 else .0
    r = c_correct / c_gold if c_gold != 0 else .0
    f = 2 * p * r / (p + r + 1e-100) if (r + p) > 1e-4 else .0

    logger.info("correct {}, predicted {}, golden {}".format(c_correct, c_predict, c_gold))

    return p, r, f

def transfor3to2(label, predt):
    all_label_d_2 = []
    all_predt_d_2 = []
    length_temp = len(predt)
    for i in range(length_temp):
        if label[i] >= 1:
            all_label_d_2.append(1)
            if predt[i] == label[i]:
                all_predt_d_2.append(1)
            else:
                all_predt_d_2.append(0)
        else:
            all_label_d_2.append(0)
            if predt[i] == label[i]:
                all_predt_d_2.append(0)
            else:
                all_predt_d_2.append(1)
    return all_label_d_2, all_predt_d_2

# calculate p, r, f1
def calculate(all_label_t, all_predt_t, all_clabel_t, epoch, printlog):
    exact_t = [0 for j in range(len(all_label_t))]
    for k in range(len(all_label_t)):
        if all_label_t[k] >= 1 and all_label_t[k] == all_predt_t[k]:
            exact_t[k] = 1
    tpi = 0 # Event pairs intra sentence correct
    li = 0  # The number of causal event pairs intra sentence
    pi = 0  # The number of causal event pairs predicted intra sentence
    tpc = 0  # Event pairs inter sentence correct
    lc = 0  # The number of causal event pairs inter sentence
    pc = 0  # The number of causal event pairs predicted inter sentence

    for i in range(len(exact_t)):

        if exact_t[i] == 1:
            if all_clabel_t[i] == 0:
                tpi += 1
            else:
                tpc += 1

        if all_label_t[i] >= 1:
            if all_clabel_t[i] == 0:
                li += 1
            else:
                lc += 1

        if all_predt_t[i] >= 1:
            if all_clabel_t[i] == 0:
                pi += 1
            else:
                pc += 1

    printlog('\tINTRA-SENTENCE:')
    recli = tpi / (li + 1e-9)
    preci = tpi / (pi + 1e-9)
    f1cri = 2 * preci * recli / (preci + recli + 1e-9)

    intra = {
        'epoch': epoch,
        'p': preci,
        'r': recli,
        'f1': f1cri
    }
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpi, pi, li))
    printlog("\t\tprecision score: {}".format(intra['p']))
    printlog("\t\trecall score: {}".format(intra['r']))
    printlog("\t\tf1 score: {}".format(intra['f1']))

    # INTER SENTENCE
    reclc = tpc / (lc + 1e-9)
    precc = tpc / (pc + 1e-9)
    f1crc = 2 * precc * reclc / (precc + reclc + 1e-9)
    cross = {
        'epoch': epoch,
        'p': precc,
        'r': reclc,
        'f1': f1crc
    }

    printlog('\tCINTER-SENTENCE:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpc, pc, lc))
    printlog("\t\tprecision score: {}".format(cross['p']))
    printlog("\t\trecall score: {}".format(cross['r']))
    printlog("\t\tf1 score: {}".format(cross['f1']))
    return tpi + tpc, pi + pc, li + lc, intra, cross