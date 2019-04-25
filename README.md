# Semantic Drift in Multilingual Representations
The repository contains code for our experiments which are described in:

Lisa Beinborn and Rochelle Choenni (2019):  
Semantic Drift in Multilingual Representations  
https://arxiv.org/pdf/1904.10820.pdf

We did not upload the extracted embeddings to github because they are too big. If you want to be sure to use exactly the same data, contact us and we will make it available.   

# Word-based experiments 
To reproduce our results, contact us to get the data and run step 3. If you want to customize the pipeline, run the following steps:
1) Download the Muse embeddings for the languages of interest from https://github.com/facebookresearch/MUSE. <br> Make sure to cite: <br> 
Conneau, Alexis, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou (2017): Word translation
without parallel data, https://arxiv.org/pdf/1710.04087.pdf. 

2) Specify data_dir to point to the folder where you store the Muse embeddings. Extract the embeddings for the test words by running extract_embeddings.py. They will be saved in *data/embeddings/word-based/*

3) Perform representational similarity analysis and clustering by running clustering.py. The results will be saved in *results/word-based/*. 

4) The plots for our qualitative examples can be reproduced by runnning the methods in the directory *detailed_analyses*.

# Sentence-based experiments
To reproduce our results, contact us to get the data and run step 2 and 3.

1) For the sentence--based experiments, you need to install Laser: https://github.com/facebookresearch/LASER. 

2) For the clustering results, run clustering_sentencebased. The results will be saved in *results/sentence-based/*.

3) For calculating the quantitative results, run calculate_treescores.py
