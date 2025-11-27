import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List


def plot_tSNE(all_hidden: List, all_labels: List[int], remove_blanks: bool = False) -> None:
    

    hidden_arr = np.vstack(all_hidden)
    labels_arr = np.array(all_labels)
    
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    hidden_2d = tsne.fit_transform(hidden_arr)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(hidden_2d[:,0], hidden_2d[:,1], c=labels_arr, cmap='tab20', s=12)
    
    plt.colorbar(scatter, label='Phoneme ID')
    plt.title("Hidden State t-SNE Visualization")
    plt.show()
    plt.savefig("tsne_plot_phonemes.png")
