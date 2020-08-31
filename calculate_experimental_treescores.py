from util.compute_tree_score import *
import argparse
import random

# This code calculates the results found in Table 2 of the paper. Make sure to set the parameter --experiment to either "word" or "sentence

# This function is used to add an additional baseline for which we scramble the distance matrix.
def permute_distance_matrix(matrix):
    n = matrix.shape[0]

    #Mask the lower part with zeros (including the diagonal)
    upper_part = np.triu(matrix,1)

    # Flatten and remove all zeros
    flattened = np.matrix.flatten(upper_part)
    no_zeros = [x for x in flattened if not x==0]

    # Permute
    random.shuffle(no_zeros)

    # Create permuted matrix, first set everything to 1 (for the diagonal)
    permuted_matrix = np.ones((n,n))
    # Fill upper part with permuted values
    permuted_matrix[np.triu_indices(n, 1)] = no_zeros

    # Make sure that lower part is symmetric
    permuted_matrix[np.tril_indices(n, -1)] = permuted_matrix.T[np.tril_indices(n, -1)]

    return permuted_matrix

def compute_score(config):
    gold = get_gold_tree(config.experiment)

    if (config.experiment == 'sentence'):

        # We obtained this score by running calculate_score_for_random_trees, we hard-code it here for reproducibility
        # calculate_score_for_random_trees()
        categories = ["rabinovich", "short", "mid", "long"]
        result_dir = "results/sentence-based/"
        name_prefix =""
        if (config.dscore):
            mean, maxi, mean_maxi = calculate_score_for_random_trees(config.experiment)

            most_distant_tree_score = maxi
            mean_random_score = mean
        else:
            # We worked with these random scores in the paper, determined over 50,000 iterations.
            mean_random_score = 2847
            most_distant_tree_score = 4596

    elif (config.experiment == 'word'):
        result_dir = "results/word-based/"
        categories = ["Pereira", "Swadesh", "combined"]
        #categories = [ "combined"]
        #categories = ["northeuralex]
        name_prefix = "ComparisonToGold_"
        if (config.dscore):
            mean, maxi, mean_maxi = calculate_score_for_random_trees(config.experiment)
            print(mean, maxi, mean_maxi)
            most_distant_tree_score = maxi
            mean_random_score = mean
        else:
            # We worked with these random scores in the paper, determined over 50,000 iterations.
            mean_random_score = 3597
            most_distant_tree_score = 5200.00



    baseline_score = mean_random_score / most_distant_tree_score

    # Initialize scrambled score, won't always be calculated
    scrambled_score = 0

    for category in categories:
        if category == "rabinovich":
            rabinovich = get_rabinovich_tree()
            score = calc_score_for_hardcoded_tree(gold, rabinovich, config.experiment)
            # We do not know the values of their distance matrix, so we cannot calculated the permuted score

        else:
            distance_matrix_name = result_dir + name_prefix +"distancematrix_" + category + ".pickle"

            with open(distance_matrix_name, 'rb') as handle:
                distance_matrix = pickle.load(handle)
                score = get_distance(distance_matrix, gold, config.experiment)

            if config.calc_scrambled_baseline:
                # Calculate scores for scrambled baseline
                scrambled_scores = []
                for i in range(1000):
                    scrambled = permute_distance_matrix(distance_matrix)
                    current_scrambled_score = get_distance(scrambled, gold, config.experiment)
                    scrambled_scores.append(current_scrambled_score)
                # Average over iterations
                scrambled_score = np.mean(np.array(scrambled_scores))



        print("Category: ", category)
        print("Distance: ", score)


        # Rabinovich et al, normalize the score  with respect to a experimentally determined "most distant tree"
        # We do not report these normalized scores in the paper because we find a comparison to the mean of randomly generated trees more plausible.
        # print("Normalize by: ", most_distant_tree_score)
        # normalized_score = score / most_distant_tree_score
        # print("Normalized score: ", normalized_score)
        # improvement = baseline_score - normalized_score
        #print("Improvement over baseline: ", improvement)

        # Compare relative to the mean of the random baseline
        notnormalized = 1 - (score / mean_random_score)
        print("Without normalization: ", notnormalized)

        # What are the scores if we scramble the distance matrix?
        if scrambled_score >0:
            print("Scrambled scores: ")
            normalized_score = scrambled_score / most_distant_tree_score
            improvement = baseline_score - normalized_score
            print("Scrambled: Improvement over baseline: ", improvement)
            notnormalized = 1 - (scrambled_score / mean_random_score)
            print("Scrambled: Without normalization: ", notnormalized)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default="word", help="Sentence or word experiment setting.")
    parser.add_argument('--dscore', type=bool, default=False, help="Re-compute the most distant tree score. ")
    parser.add_argument('--calc_scrambled_baseline', type=bool, default=False, help="Calculate the scrambled baseline scores.")
    config = parser.parse_args()

    compute_score(config)
