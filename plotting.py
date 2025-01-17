import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_values(y_list, y_label_list, best_epoch, legend_list, figsize, save_filename):
    """
    Create and save line plot of a list of loss values (per epoch).
    """
    fig, ax = plt.subplots(nrows=len(y_list), ncols=1, figsize=tuple(figsize))
    ax = ax.flatten()

    for i, (y, y_label, legend) in enumerate(zip(y_list, y_label_list, legend_list)):
        for y_i in y:
            epochs = [e + 1 for e in range(len(y_i))]
            ax[i].plot(epochs, y_i)
            ax[i].set_xticks(np.arange(1, len(y_i), 1))
        ax[i].set_ylim(bottom=min(min(y_i), 0))
        ax[i].set_xlabel('Epoch', fontsize=figsize[0])
        ax[i].axvline(x=best_epoch, color='red', linestyle='--')
        if y_label is not None:
            ax[i].set_title(y_label, fontsize=figsize[0] * 2)
        if legend is not None:
            ax[i].legend(legend, bbox_to_anchor=(1, 1), loc='upper left', prop={'size': figsize[0]})

        # Set ticks size
        ax[i].tick_params(axis='x', labelsize=figsize[0])
        ax[i].tick_params(axis='y', labelsize=figsize[0])

    plt.tight_layout(pad=2)
    plt.savefig(save_filename)
    plt.close(fig)

def plot_confusion_matrix(cm, filename, cmap='crest', fontsize=20):
    """
    Plot confusion matrix.
    """
    fig = plt.figure(figsize = (10, 7))
    sns.heatmap(cm, annot=True, annot_kws={'size': fontsize}, cmap=cmap)
    plt.ylabel('True class', fontsize=fontsize)
    plt.xlabel('Predicted class', fontsize=fontsize)
    plt.savefig(filename)
    plt.close(fig)

