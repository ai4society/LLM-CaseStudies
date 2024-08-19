# Import libraries
import numpy as np
import pandas as pd
import textdistance
import argparse

# Helper functions
def jaccard_distance(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    return 1 - textdistance.jaccard.similarity(set1, set2)

# Application
def compute_jaccard(data_path, true_col, pred_col):
    data = pd.read_csv(data_path)
    data["jaccard_dist"] = data.apply(lambda row: jaccard_distance(row[true_col], row[pred_col]), axis=1)
    
    mean_jaccard_dist = np.mean(data["jaccard_dist"])
    print(f"Mean Jaccard Distance: {mean_jaccard_dist}")
    return mean_jaccard_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the Jaccard distance between two columns in a dataset.")
    parser.add_argument('data', help="Path to the CSV file containing the dataset.")
    parser.add_argument('true_col', help="Name of the column containing the true values.")
    parser.add_argument('pred_col', help="Name of the column containing the predicted values.")
    
    args = parser.parse_args()

    compute_jaccard(args.data, args.true_col, args.pred_col)
