import os
from numpy import squeeze, mean, array, argsort

from projects.gnn_to_soz.SOZ import get_SOZ

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import time


# plot and save figures for training stats
def training_curves(train_loss, val_loss, train_acc, val_acc, log_dir):
    # set font and figures params
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Arial Narrow"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (7, 5)

    plt.figure()
    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["Training Loss", "Validation Loss"])
    plt.savefig(os.path.join(log_dir, "training curve loss.jpg"))

    plt.figure()
    plt.plot(train_acc, 'b')
    plt.plot(val_acc, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.savefig(os.path.join(log_dir, "training curve acc.jpg"))


def recall_m(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = TP / (Positives+K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = TP / (Pred_Positives+K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# evaluate and get attention
def eval_get_attn(model, test_loader, classify):
    metrics = None
    attns = []
    test_loss = []
    test_acc = []
    test_auc = []
    test_f1_score = []
    computation_times = []
    test_confusion_matrices = []
    embeddings = []
    ground_truth = []

    start_time_train = time.perf_counter()
    for b in test_loader:
        inputs, targets = b
        inputs = [inp.numpy() for inp in inputs]
        if classify == "multi":
            targets = to_categorical(targets, num_classes=3)
        else:
            targets = targets.numpy()
        # get evaluation stats
        outs = model.evaluate(inputs, targets, workers=3)
        for i, k in enumerate(model.metrics_names):
            if metrics is None:
                metrics = {k: 0 for k in model.metrics_names}
            metrics[k] += outs[i] / len(test_loader)
        test_loss.append(metrics["loss"])
        test_acc.append(metrics["accuracy"])
        test_auc.append(metrics["auc"])
        test_f1_score.append(metrics["f1_score"])

        confusion_matrix = {
            "tp": metrics["true_positives"],
            "fp": metrics["false_positives"],
            "fn": metrics["false_negatives"],
            "tn": metrics["true_negatives"],
        }
        test_confusion_matrices.append(confusion_matrix)

        epoch_time = time.perf_counter() - start_time_train
        computation_times.append(epoch_time)

        # get attention - attention is size (batch_size, num_channels, 1, num_channels)
        # squeeze and average across each column to get attention for each channel

        # FIXME attn temporary disable
        _, attn, emb = model.predict(inputs, workers=3, use_multiprocessing=True)
        attn = squeeze(attn, 2)
        attn = mean(attn, 1)
        attns.extend(attn)

        embeddings.append(emb)
        ground_truth.append(targets)


    test_stats = {
        'test_loss': test_loss[-1],
        'test_acc': test_acc[-1],
        'test_auc': test_auc[-1],
        'test_f1_score': test_f1_score[-1],
        'computation_time': computation_times[-1],
        'test_confusion': test_confusion_matrices
    }

    emb_object = {
        'embeddings': embeddings,
        'ground_truth': ground_truth
    }

    # stats on test set
    desc = "Evaluation"
    for k in model.metrics_names:
        desc_metrics = " - {}: " + ("{:.4f}" if k == "loss" else "{:.2f}")
        desc += desc_metrics.format(k, metrics[k])
    print(desc)

    return test_stats, attns, emb_object



# plot figure for attention ranking
def attention_plot(attns, ch_names, subject, log_dir):
    # normalize for each attention signal to make score between 0 and 1
    scores = [(attn - attn.min()) / (attn.max() - attn.min()) for attn in attns]
    scores = array(scores)

    # get mean and standard deviation for attention scores
    score = scores.mean(0)
    xerr = scores.std(0)
    sorter = argsort(score)
    score_sort = score[sorter]
    xerr_sort = xerr[sorter]
    chan_names_sort = [ch_names[i] for i in sorter]

    # plot figure
    plt.figure(figsize=(4, len(ch_names) // 4))
    plt.errorbar(score_sort, chan_names_sort, xerr=xerr_sort,
                 ms=5, color="firebrick", fmt="o")

    plt.title(f"Electrode ranking - {subject}")
    plt.ylabel("Electrode channels")
    plt.xlabel("Normalized attention score")
    plt.yticks(chan_names_sort)
    plt.gca().set_yticklabels(chan_names_sort, fontdict={"horizontalalignment": "right"})

    # color the tick label green if it is in the target soz
    target_soz = get_SOZ()[subject]
    for ticklabel in plt.gca().get_yticklabels():
        if ticklabel.get_text() in target_soz:
            ticklabel.set_color('g')
            ticklabel.set_weight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "rank_attn_average.pdf"), bbox_inches="tight")


# plot figure for attention ranking
def importance_plot(importances, ch_names, subject, log_dir, type=None):
    # normalize for each attention signal to make score between 0 and 1
    scores = [(impt - impt.min()) / (impt.max() - impt.min()) for impt in importances]
    scores = array(scores)

    # get mean and standard deviation for attention scores
    score = scores.mean(0)
    xerr = scores.std(0)
    sorter = argsort(score)
    score_sort = score[sorter]
    xerr_sort = xerr[sorter]
    chan_names_sort = [ch_names[i] for i in sorter]

    # plot figure
    plt.figure(figsize=(4, len(ch_names) // 4))
    plt.errorbar(score_sort, chan_names_sort, xerr=xerr_sort,
                 ms=5, color="firebrick", fmt="o")

    plt.title(f"Electrode ranking - {subject}")
    plt.ylabel("Electrode channels")
    plt.xlabel("Normalized importance score")
    plt.yticks(chan_names_sort)
    plt.gca().set_yticklabels(chan_names_sort, fontdict={"horizontalalignment": "right"})

    # color the tick label green if it is in the target soz
    target_soz = get_SOZ()[subject]
    for ticklabel in plt.gca().get_yticklabels():
        if ticklabel.get_text() in target_soz:
            ticklabel.set_color('g')
            ticklabel.set_weight('bold')

    plt.tight_layout()
    if type=="nonictal":
        plt.savefig(os.path.join(log_dir, "nonictal_rank_imptnce_average.pdf"), bbox_inches="tight")
    elif type=="ictal":
        plt.savefig(os.path.join(log_dir, "ictal_rank_imptnce_average.pdf"), bbox_inches="tight")
    else:
        plt.savefig(os.path.join(log_dir, "rank_imptnce_average.pdf"), bbox_inches="tight")
