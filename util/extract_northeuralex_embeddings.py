import pandas as pd
import pycountry
import pickle
from util.muse_utils import load_vec
import csv


# This code extracts the Muse embeddings for our test lists and saves them to save_dir
# Make sure to adjust data_dir to your local one
#data_dir = 'MUSE_DIR'
data_dir = "/Users/Lisa/Corpora/Muse_embeddings/multilingual/"

save_dir = "../data/word-based/"
name ="northeuralex"



#languages = ["sl", "bg",  "ru", "uk", "cs", "sk","hr","pl", "pt", "es", "it", "ca","fr", "ro", "nl", "de", "da","no", "sv", "en"]


with open(save_dir + "northeuralex_testing_concepts.tsv") as conceptfile:
    concepts= list(csv.reader(conceptfile, delimiter="\t"))
    languages = concepts[0][1:21]
    print(languages)
    concepts2wordforms ={}

    for line in concepts[1:]:
        key = line[0]
        values = [x for x in line[1:]]
        translations = dict(zip(languages, values))
        concepts2wordforms[key]= translations



for lang in languages:
    path = data_dir + "wiki.multi." + lang + ".vec"
    # Initialize variables
    nmax = 200000  # maximum number of word embeddings to load
    print("load embeddings for " + lang)
    embeddings, id2word, word2id = load_vec(path, nmax)
    print("loaded")
    # Get embeddings for concepts
    c_embeddings = {}
    for c, translation_dict in concepts2wordforms.items():
        try:
            wordform = translation_dict[lang].lower()
            c_id = word2id[wordform]
            c_embeddings[c] = embeddings[c_id]
        except KeyError:
            print("This should not happen")
            print(c, lang, wordform)
    print("Saving data")
    with open(save_dir + "embeddings/" + name + "_embeddings." + lang + ".pickle", 'wb') as handle:
        pickle.dump(c_embeddings, handle)


