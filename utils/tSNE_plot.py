import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List


def plot_tSNE(all_hidden: List, all_labels: List[int], remove_blanks: bool = False) -> None:
    
    if remove_blanks:
        non_blank_hidden, non_blank_labels = [], []
        for hidden, label in zip(all_hidden, all_labels):
            if label !=0:
                non_blank_labels.append(label)
                non_blank_hidden.append(hidden)
           
        all_hidden = non_blank_hidden
        all_labels = non_blank_labels

    hidden_arr = np.vstack(all_hidden)
    labels_arr = np.array(all_labels)
    
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    hidden_2d = tsne.fit_transform(hidden_arr)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(hidden_2d[:,0], hidden_2d[:,1], c=labels_arr, cmap='tab20', s=12)
    
    plt.colorbar(scatter, label='Phoneme ID')
    plt.title("Hidden State t-SNE Visualization")
    plt.savefig("tsne_plot_phonemes.png")
    plt.show()
