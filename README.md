# Semantic Drift in Multilingual Representations
The repository contains code for our experiments in:

Lisa Beinborn and Rochelle Choenni (2019):  
*Semantic Drift in Multilingual Representations*   
https://arxiv.org/pdf/1904.10820.pdf  

# Representational Similarity Analysis and Clustering
The word and sentence embeddings are too large to be uploaded to github. We stored our distance_matrices in pickle files, so that you can reproduce our plots. 

* __Word-based__: run *clustering_wordbased.py*. The results will be saved in *results/word-based/*. The plots for our qualitative examples can be reproduced by runnning the methods in the directory *detailed_analyses*.

* __Sentence-based__: run *clustering_sentencebased.py*. The results will be saved in *results/sentence-based/*.

If you want to re-run the calculations completely: 

__Word-based__
1) Download the Muse embeddings for the languages of interest from https://github.com/facebookresearch/MUSE. <br> Make sure to cite: <br> 
Conneau, Alexis, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou (2017): Word translation without parallel data, https://arxiv.org/pdf/1710.04087.pdf. 

2) Specify data_dir to point to the folder where you store the Muse embeddings. Extract the embeddings for the test words by running *extract_word_embeddings.py*. They will be saved in *data/embeddings/word-based/*


__Sentence-based__

1) The sentences have been extracted from Europarl and can be found in data/sentence-based/embeddings/part. The embeddings have been extracted using Laser. You can contact us if you want to be sure to use exactly the same embeddings. 

1) If you want to use other sentences, you need to install Laser: https://github.com/facebookresearch/LASER and follow their instructions to get embeddings. 



# Quantitative Results Translation Quality (Table 1)
1) Get the ground_truth dictionaries from https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries and put them into the folder muse_dictionaries.

2) Run *estimate_translation_quality.py*.

# Quantitative Results Clustering (Table 2) 
For calculating the quantitative evaluation of the clustering, run *calculate_experimental_treescores.py*. 

# Robustness to Translation-Induced Noise (Section 6.4)
We compared our nearest-neighbor method to using translations from the NorthEuraLex database. If you want to reproduce this, run clustering_wordbased_northeuralex.py. The quantitative results can be obtained by changing the variable category to "northeuralex" in calculate_experimental_treescores.py. 

# Requirements
We use numpy, scipy, scikit_klearn and matplotlib. If you want to extract new word embeddings, you will also need torch. Check requirements.txt for details. 
