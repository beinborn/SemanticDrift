from difflib import get_close_matches
import string

# This code is used to estimate the translation quality of the nearest neighbor method (Table 1 in paper)
def read_babelnet_output(language, data_dir):
    translation_lookup = {}

    with open(data_dir +"en_" + language + ".txt") as f:
        for line in f:
            # Adjust punctuation list
            punc = string.punctuation.replace('_', '')
            punc = punc.replace('(', '')
            punc = punc.replace(')', '')

            # Remove punctuation
            table = str.maketrans({key: None for key in punc})
            preprocessed_line = line.translate(table)

            elements = preprocessed_line.split()
            word = elements[0]
            translations = elements[1:]
            if not (translations[0]=="set()"):
                translation_lookup[word] = translations

    return translation_lookup


def read_muse_dictionaries(language, data_dir):


        translation_lookup = {}
        # There are three different files
        with open(data_dir + "en-" + language + ".txt") as f:
            for line in f:
                (key, val) = line.split()
                if key in translation_lookup.keys():
                    translation_lookup[key].append(val)
                else:
                    translation_lookup[key] = [val]
        with open(data_dir + "en-" + language + ".0-5000.txt") as f:
            for line in f:
                (key, val) = line.split()
                if key in translation_lookup.keys():
                    if (val not in translation_lookup[key]):
                        translation_lookup[key].append(val)
                else:
                    translation_lookup[key] = [val]
        with open(data_dir + "/en-" + language + ".5000-6500.txt") as f:
            for line in f:
                (key, val) = line.split()
                if key in translation_lookup.keys():
                    if (val not in translation_lookup[key]):
                        translation_lookup[key].append(val)
                else:
                    translation_lookup[key] = [val]
        return translation_lookup


languages = ["es", "fr", "it", "de", "nl", "sv", "fi", "ru", "cs", "hu", "tr", "mk", "id", "bg", "ca", "da", "et", "he",
             "no", "pl", "pt", "ro", "sk", "sl", "uk", "el", "hr"]  #

data_dir = "data/word-based/"
for name in ["Swadesh", "Pereira"]:
    print("\n\n\n" + name)
    print("Exact Close Loan Wrong Missing Quality")
    my_file = open(data_dir + name + "_MuseTranslations.csv", encoding='utf-8')
    neighbors = {}
    # Skip header
    next(my_file)
    for line in my_file:
        elems = line.split()
        key = elems[0]
        multilingual_terms = elems[1:]
        neighbors[key] = multilingual_terms

    for i, language in enumerate(languages):

        # Data
        words = sorted(neighbors.keys())
        targets = [neighbors[word][i] for word in words]
        # Only babelnet
        #translations = read_babelnet_output(language,  data_dir + "babelnet_translations/")

        # Merge Muse dictionary and babelnet
        babelnet_translations = read_babelnet_output(language, data_dir + "babelnet_translations/")
        muse_dictionaries = read_muse_dictionaries(language, data_dir + "muse_dictionaries/")

        translations = {}
        for k, v in babelnet_translations.items():
            # Add all muse translations
            if k in muse_dictionaries:
                merged = set(v)
                merged.update(muse_dictionaries[k])
                translations[k] = list(merged)

        # Add keys that are in muse but not in babelnet
        for k, v in muse_dictionaries.items():
            if not(k in translations):
                translations[k]= v


        # Count variables
        loanwords = 0
        exact_matches = 0
        close_matches = 0
        missing = 0
        incorrect = 0

        # Compare and save detailed output
        out_dir = "results/word-based/translation_quality_details/" + name
        with open(out_dir + "_" +language + ".csv", "w", encoding="utf-8") as outfile:
            for i, word in enumerate(words):
                neighbor = targets[i]
                if word in translations.keys():
                    gold_translations = translations[word]
                    if neighbor in gold_translations:
                        exact_matches += 1
                        result = "Match"
                    elif neighbor == word:
                        loanwords += 1
                        result = "Loanword"
                    else:
                        matches = get_close_matches(neighbor, gold_translations)
                        if matches != []:
                            close_matches += 1
                            result = "Found close match: " + str(matches)
                        else:
                            result = "Incorrect"
                            incorrect += 1
                else:
                    gold_translations=[]
                    missing += 1
                    result = "Not found"
                outfile.write(word + "\t" + neighbor + "\t" + result + "\t" + str(gold_translations) + "\n")

        # Output results
        all_matches = exact_matches + close_matches + loanwords
        total = len(neighbors.keys()) - missing
        quality = format(all_matches / total, '.2f')
        print(language, exact_matches, close_matches, loanwords, incorrect, missing, quality)
