import requests
import pickle
import os
def query_babelnet_synsets(word):
    # TODO Careful, do not add this to github!
    print("Querying ", word)
    key = "f31b3e82-8bce-4676-913d-437ae96fec42"
    senseurl = "https://babelnet.io/v5/getSynsetIds?"

    params = dict(lemma=word, searchLang="EN", key=key)

    # Get all synsets for the word
    resp = requests.get(url=senseurl, params=params)
    data = resp.json()
    return data

def query_babelnet_translations(id, languages):

    # TODO Careful, do not add this to github!
    key = "f31b3e82-8bce-4676-913d-437ae96fec42"
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

swadesh_file = "data/word-based/Swadesh_List.csv"
pereira_file = "data/word-based/Pereira_List.csv"
save_dir = "data/word-based/babelnet_translations/"

# Double-check which languages can be found in babelnet
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

    print(len(words), words)

    # Query translations
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

# We can only query 3 languages at a time
query_languages = languages[24:27]

# Change the start value manually if you have a limited number of requests per day
start = 218
for i in range(start, len(sorted_words)):
    word =sorted_words[i]
    print(i,word)
    synsets = word_synsets[word]

    translations_l1 = set()
    translations_l2 = set()
    translations_l3 = set()
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
    l1_file.write(word + "\t" + str(translations_l1) + "\n")
    l2_file.write(word + "\t" + str(translations_l2) + "\n")
    l3_file.write(word + "\t" + str(translations_l3) + "\n")





# TODO: quality evaluation -> do separately! But output detailed results!
    # print(word, neighbor, gold_translations)
    # if neighbor in gold_translations:
    #     # print(word, neighbor, gold_translations[word])
    #     if neighbor in gold_translations:
    #         exact_matches += 1
    #         print(word, "perfect match")
    #     elif neighbor == word:
    #         loanword += 1
    #         print(word, "loanword")
    # else:
    #     matches = get_close_matches(neighbor, gold_translations)
    #     if matches != []:
    #         close_matches += 1
    #         print(word, matches)
    #     else:
    #         not_in_dict += 1
    #         print(word, "not found")
    #
    # quality = (exact_matches + close_matches + loanword) / float(len(neighbors.keys())-not_in_dict)
    # print(l, exact_matches, close_matches, loanword, not_in_dict, quality)



