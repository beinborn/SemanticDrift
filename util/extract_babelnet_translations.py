import requests
import pickle
import os

# This code can be used to query translations from Babelnet.
# Make sure to get a babelnet key and set it accordingly.
# By default, you can only do 1000 queries a day. You might want to ask for more.

key = "YOUR_KEY"

def query_babelnet_synsets(word):
    print("Querying ", word)

    senseurl = "https://babelnet.io/v5/getSynsetIds?"
    params = dict(lemma=word, searchLang="EN", key=key)

    # Get all synsets for the word
    resp = requests.get(url=senseurl, params=params)
    data = resp.json()
    return data

def query_babelnet_translations(id, languages):

    translationurl = "https://babelnet.io/v5/getSynset?"

    # Query the synset and request its entries in the target languages
    params = dict(
        id=id,
        targetLang=[l for l in languages],
        key=key)

    resp = requests.get(url=translationurl, params=params)
    data = resp.json()

    l1 = set()
    l2 = set()
    l3 = set()

    # Iterate through all senses and extract lemma
    for sense in range(0, len(data["senses"])):
        lemma = data["senses"][sense]["properties"]["simpleLemma"].lower()
        language = data["senses"][sense]["properties"]["language"].lower()

        if language == languages[0]:
            l1.add(lemma)
        elif language == languages[1]:
            l2.add(lemma)
        elif language == languages[2]:
            l3.add(lemma)

        else:
            print("Wrong language: ", language)

    return l1,l2,l3


# Query translations for our stimuly
swadesh_file = "../data/word-based/Swadesh_List.csv"
pereira_file = "../data/word-based/Pereira_List.csv"
save_dir = "data/word-based/babelnet_translations/"

languages =["es", "fr", "it",  "de", "nl", "sv","fi", "ru", "cs", "hu",  "tr", "mk","id", "bg", "ca", "da", "et", "he", "no", "pl", "pt","ro", "sk", "sl", "uk", "el", "hr"]

synset_file = save_dir + "synsets.pickle"
if not os.path.isfile(synset_file):
    # Extract all words
    words = set()
    for word_file in [pereira_file, swadesh_file]:
        with open(word_file, "r", encoding="utf-8") as file:
            for line in file:
                word = line.split(",")[0].lower()
                words.add(word)

    # Query synsets
    word_synsets ={}
    for word in words:
        synsets = query_babelnet_synsets(word)
        word_synsets[word] = synsets

    with open(synset_file, 'wb') as handle:
            pickle.dump(word_synsets, handle)
else:
    with open(save_dir + "synsets.pickle", 'rb') as handle:
        word_synsets = pickle.load(handle)

sorted_words = sorted(word_synsets.keys())

# We can only query 3 languages at a time, adjust to your needs
query_languages = languages[0:3]

# Change the start value manually if you have a limited number of requests per day
start = 0
for i in range(start, len(sorted_words)):
    word =sorted_words[i]
    print(i,word)
    synsets = word_synsets[word]

    # Initialize counters
    translations_l1 = set()
    translations_l2 = set()
    translations_l3 = set()

    # Initialize out files
    l1_file = open(save_dir +"en_" + query_languages[0] +".txt", "a")
    l2_file = open(save_dir + "en_" + query_languages[1] + ".txt", "a")
    l3_file = open(save_dir + "en_" + query_languages[2] + ".txt", "a")

    for entry in synsets:
        id = entry["id"]
        l1, l2, l3 = query_babelnet_translations(id, query_languages)
        translations_l1.update(l1)
        translations_l2.update(l2)
        translations_l3.update(l3)
    print(translations_l1)
    print(translations_l2)
    print(translations_l3)

    # Output
    l1_file.write(word + "\t" + str(translations_l1) + "\n")
    l2_file.write(word + "\t" + str(translations_l2) + "\n")
    l3_file.write(word + "\t" + str(translations_l3) + "\n")



