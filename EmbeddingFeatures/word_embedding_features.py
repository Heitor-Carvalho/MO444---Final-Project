import os.path
import sys
import pandas as pd
import numpy as np
import multiprocessing
import pickle
import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def get_embedding_mean(tokens):
  global word_embedding
  tk_sum = np.zeros((1, 300))
  counter = 0
  for tk in tokens.split(" "):
    if tk in word_embedding.keys():
      embedded_array = [elem for elem in word_embedding[tk]]
      if len(embedded_array) != 0:
        counter += 1
        tk_sum += np.array(embedded_array).astype('float')
  return tk_sum/max(counter, 1)

def get_embedding_add(tokens):
  global word_embedding
  tk_sum = np.zeros((1, 300))
  for tk in tokens:
    if tk in word_embedding.keys():
      embedded_array = [elem for elem in word_embedding[tk]]
      if len(embedded_array) != 0:
        tk_sum += np.array(embedded_array).astype('float')
  return tk_sum

def get_embedding(tokens):
  global word_embedding
  embedded_vector = []
  for tk in tokens:
    if tk in word_embedding.keys():
      embedded_array = [elem for elem in word_embedding[tk]]
      if len(embedded_array) != 0:
        embedded_vector.append(embedded_array)
  return embedded_vector

word_embedding = load_vectors("../wiki-news-300d-1M.vec")
pool = multiprocessing.Pool(10)

FILE_TYPE = "train" # Train or test
def main():

  # Loading words and extract features
  print("Loading text")
  f_pre_processed = open(FILE_TYPE + '_pre_processed_words.pickle', 'rb')
  words_text = pickle.load(f_pre_processed)

  print("Loading bag features name")
  fd_bag_features_importance_name = open(FILE_TYPE + "_bag_words_features_name.pickle", "rb")
  bag_features_name = pickle.load(fd_bag_features_importance_name)

  print("Loading bag features")
  fd_bag_features_importance = open("_bag_words_features.pickle", "rb")
  bag_features = pickle.load(fd_bag_features_importance)

  print("Calculating weigthed embeddings of word present in the bag")
  embedded_vector_features = pool.map(get_embedding_mean, bag_features_name)
  embedded_vector_features = np.array(embedded_vector_features)
  shapes = embedded_vector_features.shape
  embedded_vector_features = np.reshape(embedded_vector_features, (shapes[0], shapes[2]))


  print("Calculating weigthed embeddings of word")
  bag_features = bag_features.toarray()
  weigthed_embedding_features = np.dot(bag_features, embedded_vector_features)

  print("Saving the embedding features XD")
  f_embbeddings = open(FILE_TYPE + 'embedding_weigthed_features.pickle', 'wb')
  pickle.dump(weigthed_embedding_features, f_embbeddings, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
  main()
