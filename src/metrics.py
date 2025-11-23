import numpy as np

def calculate_top_k_hit_rate(y_true_main, y_pred_main, k=10):
    """
    Calculates how often the actual winning numbers appear in the top K predicted numbers.

    Args:
        y_true_main: One-hot encoded actual main numbers (N, 35)
        y_pred_main: Predicted probabilities (N, 35)
        k: Top K numbers to consider

    Returns:
        float: Average hits per draw (max 5)
        float: Hit rate (percentage of draws with at least 1 hit in top k)
    """
    total_hits = 0
    at_least_one_hit_count = 0
    n_samples = len(y_true_main)

    for i in range(n_samples):
        # Get indices of actual numbers (where value is 1)
        actual_indices = np.where(y_true_main[i] == 1)[0]

        # Get indices of top k predicted numbers
        # argsort returns ascending, so we take last k and reverse
        top_k_indices = np.argsort(y_pred_main[i])[-k:][::-1]

        # Count intersection
        hits = len(np.intersect1d(actual_indices, top_k_indices))
        total_hits += hits
        if hits > 0:
            at_least_one_hit_count += 1

    avg_hits = total_hits / n_samples
    hit_rate_any = at_least_one_hit_count / n_samples

    return avg_hits, hit_rate_any

def calculate_special_accuracy(y_true_special, y_pred_special, k=3):
    """
    Calculates how often the correct special number is in the top K predictions.
    """
    hits = 0
    n_samples = len(y_true_special)

    for i in range(n_samples):
        actual_idx = np.argmax(y_true_special[i])
        top_k_indices = np.argsort(y_pred_special[i])[-k:][::-1]

        if actual_idx in top_k_indices:
            hits += 1

    return hits / n_samples
