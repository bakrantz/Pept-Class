import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix  # Just for a different way to input if needed

def visualize_confusion_matrix(confusion_matrix, class_names, filename="confusion_matrix.png"):
    """
    Visualizes a confusion matrix as a color-coded heatmap with increased annotation size
    and saves it to a file with higher resolution.

    Args:
        confusion_matrix (numpy.ndarray): The 2D confusion matrix.
        class_names (list): A list of class names (e.g., peptide names).
        filename (str, optional): The name of the file to save the plot to.
                                   Defaults to "confusion_matrix.png".
    """
    plt.figure(figsize=(8, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})  # Increase annotation font size
    plt.xlabel('Predicted Peptide', fontsize=16)
    plt.ylabel('True Peptide', fontsize=16)
    plt.title('Peptide Classification Confusion Matrix - Test Set', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi
    plt.close() # Close the plot to free up memory

if __name__ == "__main__":
    # 7x7 confusion matrices
    cm_data_1 = np.array([
        [78, 0, 0, 0, 0, 0, 0],
        [0, 75, 0, 1, 0, 2, 0],
        [0, 0, 77, 0, 1, 0, 0],
        [0, 0, 0, 78, 0, 0, 0],
        [0, 0, 0, 0, 78, 0, 0],
        [0, 0, 0, 0, 0, 79, 0],
        [0, 0, 0, 0, 0, 0, 79]
    ])

    cm_data_2 = np.array([
        [41, 3, 18, 13, 3, 0, 0],
        [4, 40, 12, 1, 8, 13, 0],
        [29, 5, 36, 1, 7, 0, 0],
        [15, 2, 2, 58, 0, 1, 0],
        [0, 11, 7, 0, 58, 2, 0],
        [0, 9, 0, 0, 1, 69, 0],
        [4, 0, 0, 0, 4, 0, 71]
    ])
    
    # Class names (assuming you have 7 peptides)
    peptide_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # Visualize the confusion matrix and save it
    visualize_confusion_matrix(cm_data_1, peptide_names, filename="./plots/cm_global_event-level_features.png")
    visualize_confusion_matrix(cm_data_2, peptide_names, filename="./plots/cm_event-level_features.png")
