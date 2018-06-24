import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

FILE_TYPE = "train" # Since we are training a model, we use only
                    # the train file

def main():

  # Loading most basic hand-crafted featutes
  fd_basic_features = open(FILE_TYPE + "_features_basic.pickle", "rb")
  basic_features_dict = pickle.load(fd_basic_features)

  features_nb = 7
  features_basic = np.zeros((len(basic_features_dict["id"]), features_nb))
  for i, id_ in enumerate(basic_features_dict["id"]):
    features_basic[i, 0] = np.abs(basic_features_dict["char_length_Q1"][i] - basic_features_dict["char_length_Q2"][i])
    features_basic[i, 1] = np.abs(basic_features_dict["word_length_Q1"][i] - basic_features_dict["word_length_Q2"][i])
    features_basic[i, 3] = np.abs(basic_features_dict["mean_word_length_Q1"][i] - basic_features_dict["mean_word_length_Q2"][i])
    features_basic[i, 4] = basic_features_dict["similar_words_ratio"][i]
    features_basic[i, 5] = basic_features_dict["similar_tags_ratio"][i]
    features_basic[i, 6] = basic_features_dict["longest_common_sequence"][i]
    features_basic[i, 7] = basic_features_dict["longest_common_sequence_tags"][i]

  features_basic[np.isnan(features_basic)] = 0
  labels = np.array(basic_features_dict["label"]).flatten()

  # Loading most featutes from word embedding
  fd_embedding_features = open(FILE_TYPE + "_embedding_weigthed_features.pickle", "rb")
  embedding_features = pickle.load(fd_embedding_features)

  full_features = np.concatenate((features_basic, embedding_features), axis=1)

  random_forest = RandomForestClassifier(max_depth = 60, n_estimators = 40)
  random_forest.fit(full_features, labels)
  labels_predict = random_forest.predict(full_features)
  cm = confusion_matrix(labels, labels_predict)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("Let's check the confusion matrix:")
  print(cm)

  print("Let's get the scores!")
  print("Calculated accuracy score: %f" % metrics.accuracy_score(labels_predict, labels))
  print("Calculated log_loss score: %f" % metrics.log_loss(labels_predict, labels))

  # Saving model to test
  fd_random_forest_model = open(FILE_TYPE + "_random_forest_final_mode.pickle", "wb")
  pickle.dump(random_forest, fd_random_forest_model, protocol=pickle.HIGHEST_PROTOCOL)









if __name__ == '__main__':
  main()
