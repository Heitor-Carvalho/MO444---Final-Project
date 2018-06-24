from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle
import numpy as np

FILE_TYPE = "train" # Since we are training a model, we use only
                    # the train file type option

def main():
  feature_fd = open(FILE_TYPE + "_features_basic.pickle", "rb")
  features_dict = pickle.load(feature_fd)

  features_nb = 7
  features = np.zeros((len(features_dict["id"]), features_nb))
  for i, id_ in enumerate(features_dict["id"]):
    features[i, 0] = np.abs(features_dict["char_length_Q1"][i] - features_dict["char_length_Q2"][i])
    features[i, 1] = np.abs(features_dict["word_length_Q1"][i] - features_dict["word_length_Q2"][i])
    features[i, 2] = np.abs(features_dict["mean_word_length_Q1"][i] - features_dict["mean_word_length_Q2"][i])
    features[i, 3] = features_dict["similar_words_ratio"][i]
    features[i, 4] = features_dict["similar_tags_ratio"][i]
    features[i, 5] = features_dict["longest_common_sequence"][i]
    features[i, 6] = features_dict["longest_common_sequence_tags"][i]

  features[np.isnan(features)] = 0
  labels = np.array(features_dict["label"]).flatten()

  print("Percentagen of %f - Classe +" % float(np.sum(labels == 1)/len(labels)))
  print("Percentagen of %f - Classe -" % float(np.sum(labels == 0)/len(labels)))

  # Using LogisticRegression
  reg_options = [1e-5, 1e-3, 1, 100, 1e3, 1e5]
  for reg in reg_options:
    logistic = LogisticRegression(C=reg)
    scores = cross_val_score(logistic, features, labels, cv=5)
    print("Score for reg %f" % reg)
    print("Scores:")
    print(scores)
    logistic.fit(features, labels)
    cm = confusion_matrix(labels, logistic.predict(features))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Let's check the normalized confusion_matrix - LogisticRegression")
    print(cm)

  # Using Random Forest
  random_forest = RandomForestClassifier(max_depth = 100, n_estimators = 10, random_state=0, class_weight="balanced")
  scores = cross_val_score(random_forest, features, labels, cv=5)
  random_forest.fit(features, labels)
  labels_predict = random_forest.predict(features)
  cm = confusion_matrix(labels, labels_predict)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("Let's check the normalized confusion_matrix")
  print(cm)

  print("Let's check some the feature importance")
  print("Features importances: [diff_char_length, diff_word_length, mean_word_length_diff, similar_tags_ratio, \
                                 similar_tags_ratio, longest_common_sequence, longest_common_tags] ")
  print(random_forest.feature_importances_)
  print("Let's get the scores!")
  print("Calculated accuracy score: %f" % metrics.accuracy_score(labels_predict, labels))
  print("Calculated log_loss score: %f" % metrics.log_loss(labels_predict, labels))


if __name__ == '__main__':
  main()
