import pickle
import numpy as np
from scipy import spatial
from sklearn import preprocessing
import os
# This code runs similarity search on the chosen sentences to check the quality of the embeddings
data_dir = '../data/sentence-based/embeddings/part/'
save_dir = "../results/similarity_search?"
os.makedirs(save_dir, exist_ok=True)

# Get category: short-> 0-200, medium -> 200-400, long -> 400-600
start = 0
end = 200
dim = 1024

target_langs = ["de", "nl", "fr", "es", "bg", "da", "cs", "pl", "pt", "it", "ro", "sk", "sl", "sv", 'lt', 'lv']

# Get embeddings of the english sentences 
en_embeddings = np.fromfile(data_dir + "en_embeddings.raw", dtype=np.float32, count=-1)
en_embeddings.resize(en_embeddings.shape[0] // dim, dim)
en_embeddings = en_embeddings[start:end]
en_embeddings = preprocessing.normalize(en_embeddings, norm="l2")

# Get embeddings of the target languages
target_embeddings = []
for l in target_langs:
    X = np.fromfile(data_dir + l + "_embeddings.raw", dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)
    X = X[start:end]
    target_embeddings.append(X)

# Get english sentences 
with open(data_dir + "en_sents.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]
content = content[start:end]

incorrect = []
for l in range(len(target_langs)):
    # Get corresponding sentences in the target language
    with open(data_dir + target_langs[l] + "_sents.txt") as f:
        cont_targ = f.readlines()
    cont_targ = [x.strip() for x in cont_targ]
    cont_targ = cont_targ[start:end]

    correct_counter = 0
    n = 0
    incorrect_sent_ids = []
    embeds = target_embeddings[l]
    embeds = preprocessing.normalize(embeds, norm="l2")
    file = open(save_dir + "nn_in_" + target_langs[l] + "_long_incorrect.txt", "w")
    # For each english sentence get nearest neighbour
    for i in range(len(en_embeddings)):
        en = en_embeddings[i]
        dist = []
        for j in range(len(embeds)):
            s = spatial.distance.cosine(en, embeds[j])
            dist.append(s)
        ind = np.argmin(dist)
        n += 1
        # Nearest neighbour in language l is the translation of the sentence in English.
        if (i == ind):
            correct_counter += 1
        # Else, write incorrect nearest neighbours to a text file
        else:
            incorrect_sent_ids.append(n)
            file.write('T' + str(n) + ': ' + content[i] + "\n")
            file.write('NN' + str(n) + ': ' + content[ind] + "\n")
            file.write("NN T" + str(n) + ': ' + cont_targ[ind] + "\n")
    file.close()
    incorrect.append(incorrect_sent_ids)
    print("Target lang: ", target_langs[l], ", Identical: ", correct_counter, ", Percentage: ",
          correct_counter / len(content))

# Dump all ids of incorrect sentences 
with open(save_dir + "incorrect_sentences.pickle", 'wb') as handle:
    pickle.dump(incorrect, handle)

# print(set.intersection(*map(set,incorrect)))
