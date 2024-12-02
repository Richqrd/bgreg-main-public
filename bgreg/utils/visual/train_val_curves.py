import pickle
import matplotlib.pyplot as plt


def training_curves(train_loss, val_loss, train_acc, val_acc):
    # FIXME: parameterize this function to account for any patient/file
    # set font and figures params
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Arial Narrow"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (5, 4)

    plt.figure()
    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["Training Loss", "Validation Loss"])

    ax = plt.gca()
    ax.set_ylim(0.5, 0.8)
    plt.savefig("pt6_training_curve_loss.pdf", bbox_inches='tight', pad_inches=0.1)

    plt.figure()
    plt.plot(train_acc, 'b')
    plt.plot(val_acc, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    ax.set_ylim(0.5, 0.8)
    plt.savefig("pt6_training_curve_val.pdf", bbox_inches='tight', pad_inches=0.1)


pt6_statsfile = "HPC-results-patience/pt6/nf-ones_ef-ones_adj-corr_link-0.3/train_val_stats.pickle"
pt11_statsfile = "HPC-results-patience/pt11/nf-ones_ef-ones_adj-corr_link-0.3/train_val_stats.pickle"

with open(pt6_statsfile, "rb") as fpt6:
    pt_stats = pickle.load(fpt6)
    train_loss = pt_stats["train_loss"]
    val_loss = pt_stats["val_loss"]
    train_acc = pt_stats["train_acc"]
    val_acc = pt_stats["val_acc"]
    training_curves(train_loss, val_loss, train_acc, val_acc)