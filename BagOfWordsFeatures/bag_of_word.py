from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import pickle

FILE_TYPE = "train" # or test

def main():
  count_vect = CountVectorizer(ngram_range=(1, 2), max_features=5000)
  tf_idf_transformer = TfidfTransformer()

  f_pre_processed = open(FILE_TYPE + "_pre_processed_words.pickle", 'rb')
  words_text = pickle.load(f_pre_processed)


  questions_joined = [q1 + q2 for q1, q2 in zip(words_text["questions1_stemmed"], words_text["questions2_stemmed"])]

  questions_joined_count = count_vect.fit_transform(questions_joined)
  tf_idf_matrix = tf_idf_transformer.fit_transform(questions_joined_count)

  f_bag_features = open(FILE_TYPE + "_bag_words_features.pickle", 'wb')
  pickle.dump(tf_idf_matrix, f_bag_features, protocol=pickle.HIGHEST_PROTOCOL)

  features_name = count_vect.get_feature_names()
  f_bag_features_name = open(FILE_TYPE + "_bag_words_features_name.pickle", 'wb')
  pickle.dump(features_name, f_bag_features_name, protocol=pickle.HIGHEST_PROTOCOL)

  # After some tests, this information was not used in the final model,
  # since SVD did not help us to get a better classifier
  svd = TruncatedSVD(n_components=500)
  normalizer = Normalizer(copy=False)
  lsa = make_pipeline(svd, normalizer)
  svd_bag = lsa.fit_transform(tf_idf_matrix)

  f_bag_features_svd = open(FILE_TYPE + "_bag_words_features_svd.pickle", 'wb')
  pickle.dump(svd_bag, f_bag_features_svd, protocol=pickle.HIGHEST_PROTOCOL)
  print("Explained variance ratio: %f" % svd.explained_variance_ratio_.sum())












if __name__ == '__main__':
  main()
