import pickle
import os
from util.muse_utils import load_vec, get_nn


# This code extracts the Muse embeddings for our test lists and saves them to save_dir
# Make sure to adjust data_dir to your local one
data_dir = '/Users/lisa/Corpora/embeddings/multilingual/'
en_path = data_dir + "wiki.multi.en.vec"
save_dir = "data/word-based"

# Initialize variables
nmax = 200000  # maximum number of word embeddings to load
src_embeddings, src_id2word, src_word2id = load_vec(en_path, nmax)
words_en = []
vectors_en = []
swadesh_file = "data/Swadesh_List.csv"
pereira_file = "data/Pereira_List.csv"

# We exclude arabic (because the provided file has a technical problem) and vietnamese (because the quality seems off)
languages =["es", "fr", "it",  "de", "nl", "sv","fi", "ru", "cs", "hu",   "tr", "mk","id", "bg", "ca", "da", "et", "he", "no", "pl", "pt","ro", "sk", "sl", "uk", "el", "hr"]


# Get test words
for word_file in [pereira_file, swadesh_file]:
    en_data = {}
    with open(word_file, "r", encoding="utf-8") as file:
        name = os.path.basename(word_file).split("_")[0]
        print(name)
        next(file)
        for line in file:
            word = line.split(",")[0].lower()
            print(word)
            try:
                # Get English embeddings
                vector = src_embeddings[src_word2id[word]]
                vectors_en.append(vector)
                words_en.append(word)
                en_data[word] = vector
            except KeyError:
                print(word + " not in vectors")

    # Save English embeddings
    with open(save_dir + "embeddings/" + name + "_embeddings.en.pickle", 'wb') as handle:
        pickle.dump(en_data, handle)

    # Get nearest neighbour in L2 for each test word
    l2_vectors = []
    l2_words = []
    for lang in languages:
        l2_path = data_dir + "wiki.multi." + lang + ".vec"
        l2_embeddings, l2_id2word, l2_word2id = load_vec(l2_path, nmax)
        vectors_l2 = {}
        words_l2 = []

        print("Language: " + lang)
        for word in words_en:
            k_best, scores, tgt_id2word = get_nn(word, src_embeddings, src_id2word, l2_embeddings, l2_id2word, K=3)
            nearest_l2_neighbour = tgt_id2word[k_best[0]]
            words_l2.append(nearest_l2_neighbour)
            vectors_l2[word] = l2_embeddings[l2_word2id[nearest_l2_neighbour]]

        l2_vectors.append(vectors_l2)
        l2_words.append(words_l2)

        print("Saving data for language")
        with open(save_dir + "embeddings/" + name + "_embeddings." + lang + ".pickle", 'wb') as handle:
            pickle.dump(vectors_l2, handle)

    # Save all translations in one file for quality checks
    with open(save_dir+ name + "_MuseTranslations.csv", "w",
              encoding="utf-8") as out:
        for i in range(0, len(words_en)):
            out.write(words_en[i] + "\t")
            for k in range(0, len(languages)):
                out.write(l2_words[k][i] + "\t")
            out.write("\n")
