import pandas as pd
import pycountry
import pickle
from util.muse_utils import load_vec
import unidecode
# Download northeuralex list from http://northeuralex.org/download (Lexical Data)
northeuralexfile = "../data/word-based/northeuralex-0.9-forms.tsv"

# This code extracts the Muse embeddings for our test lists and saves them to save_dir
# Make sure to adjust data_dir to your local one
#data_dir = 'MUSE_DIR'
data_dir = "/Users/Lisa/Corpora/Muse_embeddings/multilingual/"

save_dir = "../data/word-based/"
name ="northeuralex"

# check if at least one word in the list contains a whitespace
def contains_multiwords(wordlist):
    for word in wordlist:
        elems = word.split(" ")
        if len(elems)>1:
            print(word)
            return True
    return False

# --------------------------------------------------------
# Extracting northeuralex translations for our languages
with open(northeuralexfile, "r", encoding="utf-8") as wordlist:

    northeuralex = pd.read_csv(wordlist, keep_default_na=False, sep="\t", error_bad_lines=False)
    translations = northeuralex[["Language_ID", "Concept_ID", "Word_Form"]]

    languages = ["sl", "bg", "ru", "uk", "cs", "sk","hr","pl", "pt", "es", "it", "ca","fr", "ro", "nl", "de", "da","no", "sv", "en"]

    # map languages into 3-char codes
    language_mapping ={}
    for lang in languages:
        language_mapping[lang] = pycountry.languages.get(alpha_2=lang).alpha_3


    english_subset = translations[translations.Language_ID=="eng"]
    # Concepts_IDs are in German, but our word lists are in English, create a mapping
    northeuralex_concepts= dict(zip(english_subset.Word_Form, english_subset.Concept_ID))

    target_languages = translations[translations.Language_ID.isin(list(language_mapping.values()))]

# --------------------------------------------------------
# Removing all test concepts that are translated into more than one word in one of the languages
    concepts2wordforms ={}
    ignored_concepts =[]
    for concept_id in northeuralex_concepts.values():
        concept_info = target_languages[target_languages.Concept_ID == concept_id]
        word_forms = dict(zip(concept_info.Language_ID, concept_info.Word_Form))

        # Check if concept is expressed in two words in any of the languages
        if not contains_multiwords(list(word_forms.values())):
            concepts2wordforms[concept_id] = word_forms
        else:
            print("Ignoring multiword concept:" +  concept_id)
            ignored_concepts.append(concept_id)

    print("Included concepts: " + str(len(concepts2wordforms.keys())))
    print("Ignored concepts: " + str(len(ignored_concepts)))

# --------------------------------------------------------
# Removing all test concepts that are not available in Muse for one of the languages
    testing_concepts =list(concepts2wordforms.keys())
    translations ={}
    for lang in languages:
        muse_terms ={}
        path = data_dir + "wiki.multi." + lang + ".vec"
        # Initialize variables
        nmax = 200000  # maximum number of word embeddings to load
        print("load embeddings for " + lang)
        embeddings, id2word, word2id = load_vec(path, nmax)
        print("loaded")
        # Get embeddings for concepts
        for c, translation_dict in concepts2wordforms.items():
            c_embeddings = {}
            try:
                wordform = translation_dict[language_mapping[lang]].lower()
                c_muse_id = word2id[wordform]
                muse_terms[c]= wordform

            except KeyError:
                print("\nConcept not available in Muse: ")
                print(c, lang, wordform)
                # For Slovenian, there was a big problem with accents. I am not sure if this is the way to solve it.

                unaccented_wordform = unidecode.unidecode(wordform)
                print("Trying without accents: " + unaccented_wordform)
                try:
                    c_muse_id = word2id[unaccented_wordform]
                    print("That worked")
                    muse_terms[c]= unaccented_wordform

                except KeyError:
                    print("Also not working")
                    print("Removing it from testing concepts")
                    if c in testing_concepts:
                        testing_concepts.remove(c)
        translations[lang]= muse_terms

        print("Concepts for testing: " + str(len(testing_concepts)))
        print(testing_concepts)

    # --------------------------------------------------------
    # Save testing concepts and translations to file

    with open(save_dir + "northeuralex_testing_concepts.tsv", 'w') as outfile:
        outfile.write("Concept_Id\t")
        for l in languages:
            outfile.write(l)
            outfile.write("\t")
        outfile.write("\n")
        for c in testing_concepts:
            outfile.write(c)
            outfile.write("\t")
            for l in translations.keys():
                translation = translations[l][c]
                outfile.write(translation)
                outfile.write("\t")
            outfile.write("\n")



# Old stuff, can probably deleted, keep it for the moment.

        # swadesh_file = "../data/word-based/Swadesh_List.csv"
        # swadesh_words =[]
        # with open(swadesh_file, "r", encoding="utf-8") as file:
        #     # skip header
        #     next(file)
        #     for line in file:
        #         swadesh_words.append(line.split(",")[0].lower())
        # swadesh_concept_ids = []
        # for w in swadesh_words:
        #     try:
        #         swadesh_concept_ids.append(en2concepts[w])
        #     except KeyError:
        #         print("Not in Northeuralex: " + w)
        # print(len(swadesh_concept_ids), swadesh_concept_ids)
        # subset = translations[(translations["Language_ID"]==language_mapping[lang]) & (translations["Concept_ID"].isin(swadesh_concept_ids))]
        # # TODO: are words sorted by Concept_ID?
        # words = list(subset["Word_Form"])
        # print(words)
        # break
        # ids = [word2id[word.lower()] for word in words]
        # embeddings= [embeddings[id] for id in sorted(ids)]
        # print("Saving data for language")
        # with open(save_dir + "embeddings/" + name + "_embeddings." + lang + ".pickle", 'wb') as handle:
        #     pickle.dump(embeddings, handle)



    # print(en2concepts)
    #
    # swadesh_file = "../data/word-based/Pereira_List.csv"
    # errors =0
    # concepts =[]
    # with open(swadesh_file, "r", encoding="utf-8") as file:
    #     # skip header
    #     next(file)
    #     for line in file:
    #         word = line.split(",")[0].lower()
    #         try:
    #             concept =  en2concepts_lowercase[word]
    #             concepts.append(concept)
    #         except KeyError:
    #             print("Not in northeuralex: " + word)
    #             errors +=1
    # #print("Number of missing concepts in northeuralex: " + str(errors))
    # # Swadesh: 28
    # # Northeuralex: 114
    #
    # # All languages

    #
    #     print(code)
    #     subset = translations[translations.Language_ID==code]
    #
    #     for concept in concepts:
    #         translation = subset[subset.Concept_ID==concept]
    #         print(concept, translation.Word_Form.values)
    #         break
    #
