import pickle
import numpy as np
from random import shuffle
from util.evaluate_RSA import get_dists, compute_distance_over_dists
from util.compute_tree_score import *
import matplotlib.pyplot as plt
from util.external_laser import laser_embed
import os
import sys
def get_embeddings(sample, language_code):
    # TODO: adjust model dir
    model_dir = "/Users/lisa/PycharmProjects/LASER-master/models/"
    os.environ['LASER'] = model_dir
    encoder = model_dir + "bilstm.93langs.2018-12-26.pt"
    bpe_codes = model_dir + "93langs.fcodes"
    dim = 1024
    output_file = os.getcwd() + "/data/sentence-based/embeddings/full/" + language_code + "_embeddings.raw"

    if not os.path.isfile(output_file):
        print(sample[0], language_code)
        tempfile = os.getcwd() + "/data/tempfile.txt"
        # This is very messy fiddling with inpt and output
        with open(tempfile, "w") as text_file:
            text_file.write("\n".join(sample))
        laser_embed.main(["--encoder",encoder, "--token-lang", language_code, "--bpe-codes", bpe_codes, "--output", output_file, "--ifname", tempfile])

    X = np.fromfile(output_file, dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)
    return X


languages = ["en", "de", "nl", "fr", "es", "bg", "da", "cs", "pl", "pt", "it", "ro", "sk", "sl", "sv", 'lt', 'lv']
data_dir = "data/sentence-based/"
result_dir = "results/sentence-based/learning_curve/"
with open(data_dir + "translation_dict.pckl", 'rb') as handle:
    translations = pickle.load(handle)

# Store sentence ids and sentence length in dictionaries
id = 0
sentence_ids = {}
sentence_lengths = {}

# Assign ids to sentences and store sentence length
# Sentence lengths are rounded to the nearest 10,
# e.g. sentences with length 16-20 are all stored as 20, length 10-15 is stored as 10

for sent in translations.keys():
    sentence_ids[id] = sent
    length = round(len(sent.split(" ")), -1)
    # Not enough super long sentences, skip them
    if length <=60 and length >0:
        if length in sentence_lengths.keys():
            length_ids = sentence_lengths[length]
            length_ids.append(id)
        else:
            length_ids = [id]

        sentence_lengths[length] = length_ids
        id += 1

lengths = sorted(sentence_lengths.keys())

# Get English embeddings
en_sentences = [sentence_ids[id] for id in sorted(sentence_ids.keys())]
print("Number of sentences")
print(len(en_sentences))
print("Number of translations")
print(len(list(translations.values())))
en_embeddings = get_embeddings(en_sentences, "en")
print(en_embeddings.shape)
embeddings = [en_embeddings]

# Get embeddings for other languages
for k in range(1, len(languages)):
    lang_sentences = [translations[sentence_ids[id]][k-1] for id in sorted(sentence_ids.keys())]
    lang_embeddings = get_embeddings(lang_sentences, languages[k])
    embeddings.append(lang_embeddings)

num_iterations = 50
sample_size = 200
gold = get_gold_tree()
most_distant_tree_score = 4520

# Iterate through sentence lengths
mean_scores =[]
std_scores =[]
measured_lengths =[]
for l in lengths:
    # Repeat for num_iterations
    scores =[]
    if len(sentence_lengths[l])>sample_size:
        measured_lengths.append(l)
        for i in range(0, num_iterations):

            # Extract random sample
            sents = sentence_lengths[l]

            shuffle(sents)

            sample = sents[0:sample_size]

            sample_embeddings =[]
            for k in range(0, len(languages)):
                sample_embeddings.append(np.asarray(embeddings[k])[sample][:])


            print("Starting RSA")
            # Apply RSA
            # Calculate representational similarity analysis
            x, C = get_dists(sample_embeddings, labels=languages, ticklabels=[], distance="cosine", save_dir=result_dir)
            distance_matrix = compute_distance_over_dists(x, C, languages, save_dir=result_dir)

            score = get_distance(distance_matrix, gold)
            norm_score = score / most_distant_tree_score
            print(l, i, norm_score)
            scores.append(norm_score)
        mean = np.mean(np.asarray(scores))
        std = np.std(np.asarray(scores))

        mean_scores.append(mean)
        std_scores.append(std)

mean_scores = np.asarray(mean_scores)
std_scores = np.asarray(std_scores)
plt.figure()
print(measured_lengths)
print(mean_scores)
print(std_scores)
# Results for 10 iterations
# measured_lengths = np.asarray([10, 20, 30, 40, 50, 60])
# mean_scores = np.asarray([0.47088496, 0.45663717, 0.42017699, 0.43575221, 0.47681416, 0.56964602])
# std_scores = np.asarray([0.12850853, 0.11157113 ,0.07924362, 0.08269124, 0.10506613, 0.10881433])
plt.xlabel("Sentence Length")
plt.ylabel("Mean Tree Score")
plt.ylim(0,1)
plt.fill_between(measured_lengths, mean_scores-std_scores,mean_scores + std_scores, alpha=0.1,  color="b")
plt.plot(measured_lengths, mean_scores, 'o:', color="b")
plt.savefig(result_dir +"learningCurve.png")

results =[measured_lengths, mean_scores, std_scores]
with open(result_dir +"learningcurveresults.txt", 'w') as out:
    out.write("Length\tMeanTreeScore\tStdTreeScore")
    for x in zip(*results):
        out.write("{0}\t{1}\t{2}\n".format(*x))



plt.show()

