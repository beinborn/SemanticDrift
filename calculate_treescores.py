from util.compute_tree_score import *


gold = get_gold_tree()
rabinovich = get_rabinovich_tree()

rabinovich_score = calc_score_for_hardcoded_tree(gold, rabinovich)

# We obtained this score by running calculate_score_for_random_trees, we hard-code it here for reproducibility
#calculate_score_for_random_trees()
most_distant_tree_score = 4520


print("Rabinovich Score: ", rabinovich_score)
print("Rabinovich Normalized score: ", rabinovich_score / most_distant_tree_score)

result_dir = "results/sentence-based/"
for category in ["short", "mid", "long"]:
    distance_matrix_name = result_dir + "distancematrix_" + category + ".pickle"
    with open(distance_matrix_name, 'rb') as handle:
        distance_matrix = pickle.load(handle)
    score = get_distance(distance_matrix, gold)
    print("Category: " + category)
    print("Score: ", score)
    print("Normalized score: ")
    print("%.3f" % (score/most_distant_tree_score))







