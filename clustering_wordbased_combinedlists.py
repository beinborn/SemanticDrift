import os
import pickle


from util.evaluate_RSA import get_dists, compute_distance_over_dists

data_dir = "data/word-based/embeddings/"
save_dir = "results/word-based/"
os.makedirs(save_dir, exist_ok=True)


# All languages
# languages = ["it", "pt", "es", "ca", "fr", "en", "nl", "de", "sv", "da", "no", "fi", "et", "ru", "uk", "mk", "bg", "ro",
#             "pl", "cs", "sk", "sl", "hr", "hu", "el", "he", "tr", "id"]


# Languages for the gold tree
languages = ["it", "pt", "es", "ca", "fr", "en", "nl", "de", "sv", "da", "no", "ru", "uk", "bg", "ro", "pl", "cs", "sk",
             "sl", "hr"]
print(len(languages))

words = set()
vectors =[]

for lang in languages:
    name = "swadesh"
    with open(data_dir + name + "_embeddings." + lang + ".pickle", 'rb') as handle:
        swadesh_dict = pickle.load(handle)
        if len(words) == 0:
            words.add(swadesh_dict.keys())

    name = "pereira"
    with open(data_dir + name + "_embeddings." + lang + ".pickle", 'rb') as handle:
        pereira_dict = pickle.load(handle)
        if len(words) < 300:
            words.add(pereira_dict.keys())

    combined_dict = {**swadesh_dict, **pereira_dict}
    print(combined_dict)

    # get vectors
    langvectors = []
    for word in words:
        langvectors.append(combined_dict[word])
    vectors.append(langvectors)

### Representational Similarity Analysis
# Calculate similarity matrices for all languages
x, C = get_dists(vectors, labels=languages, ticklabels=words, distance="cosine", save_dir=save_dir + name)

# Calculate RSA over all languages
distance_matrix = compute_distance_over_dists(x, C, languages, save_dir=save_dir + name)

#
# Save distance matrix
with open(save_dir + "ComparisonToGold_distancematrix_combined.pickle", 'wb') as handle:
    pickle.dump(distance_matrix, handle)

