import io
import numpy as np
# Helper methods from https://github.com/facebookresearch/MUSE/blob/master/demo.ipynb

# Load the vectors
def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

# Get the nearest neighbour in the target embeddings for a word in the source embeddings
def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    word2id = {v: k for k, v in src_id2word.items()}

    # Get the source word embedding
    word_emb = src_emb[word2id[word]]

    # Calculate cosine distance of all words in the target embeddings
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))

    # Return the k-nearest neighbours
    k_best = scores.argsort()[-K:][::-1]
    return k_best, scores, tgt_id2word

# # Quality check: nearest Russian neighbours of "small"
# word ="small"
#
# # Load English and Russian embeddings
# #data_dir = 'MUSE_DIR'
# data_dir = '/Users/lisa/Corpora/Muse_embeddings/multilingual/'
# en_path = data_dir + "wiki.multi.en.vec"
#
# nmax = 200000  # maximum number of word embeddings to load
#
# src_embeddings, src_id2word, src_word2id = load_vec(en_path, nmax)
# vector = src_embeddings[src_word2id[word]]
#
# l2_path = data_dir + "wiki.multi.ru.vec"
# l2_embeddings, l2_id2word, l2_word2id = load_vec(l2_path, nmax)
#
# # Get nearest neighbors
# k =10
# k_best, scores, tgt_id2word = get_nn(word, src_embeddings, src_id2word, l2_embeddings, l2_id2word, K=k)
#
# for i, idx in enumerate(k_best):
#     print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))

