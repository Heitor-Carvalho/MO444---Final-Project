import pickle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np

FILE_TYPE = "train" # Since we are training a model, we use only
                    # the train file type option

def main():
  fd_basic_feat = open(FILE_TYPE + "_features_basic.pickle", "rb")
  basic_features = pickle.load(fd_basic_feat)
  fd_bag_features = open(FILE_TYPE + "_bag_words_features.pickle", "rb")
  bag_feat = pickle.load(fd_bag_features)

  fd_bag_features_name = open(FILE_TYPE + "_bag_words_features_name.pickle", "rb")
  bag_feat_name = pickle.load(fd_bag_features_name)

  labels = basic_features["label"]

  random_forest = RandomForestClassifier(max_depth = 60, n_estimators = 40)
  random_forest.fit(bag_feat, labels)
  labels_predict = random_forest.predict(bag_feat)
  cm = confusion_matrix(labels, labels_predict)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("Let's check the confusion matrix:")
  print(cm)

  print("Let's get the scores!")
  print("Calculated accuracy score: %f" % metrics.accuracy_score(labels_predict, labels))
  print("Calculated log_loss score: %f" % metrics.log_loss(labels_predict, labels))



if __name__ == '__main__':
  main()
