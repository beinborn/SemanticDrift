from util.compute_tree_score import *
import argparse

def compute_score(config):
    gold = get_gold_tree(config.experiment)

    if (config.experiment == 'sentence'):
        rabinovich = get_rabinovich_tree()

        rabinovich_score = calc_score_for_hardcoded_tree(gold, rabinovich, config.experiment)

        # We obtained this score by running calculate_score_for_random_trees, we hard-code it here for reproducibility
        # calculate_score_for_random_trees()
        most_distant_tree_score = 4520

        print("Rabinovich Score: ", rabinovich_score)
        print("Rabinovich Normalized score: ", rabinovich_score / most_distant_tree_score)

        result_dir = "results/sentence-based/"
        for category in ["short", "mid", "long"]:
            distance_matrix_name = result_dir + "distancematrix_" + category + ".pickle"
            with open(distance_matrix_name, 'rb') as handle:
                distance_matrix = pickle.load(handle)
            score = get_distance(distance_matrix, gold, config.experiment)
            print("Category: " + category)
            print("Score: ", score)
            print("Normalized score: ")
            print("%.3f" % (score / most_distant_tree_score))

    elif (config.experiment == 'word'):
        if (config.dscore):
            mean, maxi, mean_maxi = calculate_score_for_random_trees(config.experiment)
            most_distant_tree_score = maxi
        else:
            most_distant_tree_score = 5352

        result_dir = "results/word-based/"

        for category in ["Pereira", "Swadesh"]:
            distance_matrix_name = result_dir + "ComparisonToGold_distancematrix_" + category + ".pickle"
            with open(distance_matrix_name, 'rb') as handle:
                distance_matrix = pickle.load(handle)
            score = get_distance(distance_matrix, gold, config.experiment)
            print("Category: " + category)
            print("Score: ", score)
            print("Normalized score: ")
            print("%.3f" % (score / most_distant_tree_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default="word", help="Sentence or word experiment setting.")
    parser.add_argument('--dscore', type=bool, default=False, help="Re-compute the most distant tree score. ")

    config = parser.parse_args()

    compute_score(config)
