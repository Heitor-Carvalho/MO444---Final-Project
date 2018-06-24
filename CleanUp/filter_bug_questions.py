import pandas as pd

TRAIN_CSV_FILE = "train.csv"

def main():
  train_data = pd.read_csv(TRAIN_CSV_FILE)
  train_data = train_data.dropna(axis=0, how='any')

  ids = train_data["id"].tolist()
  q1ids = train_data["qid1"].tolist()
  q2ids = train_data["qid2"].tolist()
  question1 = train_data["question1"].tolist()
  question2 = train_data["question2"].tolist()
  label = train_data["is_duplicate"].tolist()

  min_length_valid_question = 6;
  ids_ok = []
  q1ids_ok = []
  q2ids_ok = []
  q1question_ok = []
  q2question_ok = []
  label_ok = []
  for i in range(0, len(ids)):
    q1 = question1[i]
    q2 = question2[i]
    if(not isinstance(q1, basestring) or not isinstance(q2, basestring)):
      q1 = str(q1)
      q2 = str(q2)
      print "Removing bug question id %d" % ids[i]
      print "Q1: ", q1
      print "Q2: ", q2
      continue
    if(len(q1) < min_length_valid_question or len(q2) < min_length_valid_question):
      print "Removing bug question id %d" % ids[i]
      print "Q1: ", q1
      print "Q2: ", q2
      continue
    ids_ok.append(ids[i])
    q1ids_ok.append(q1ids[i])
    q2ids_ok.append(q2ids[i])
    q1question_ok.append(question1[i])
    q2question_ok.append(question2[i])
    label_ok.append(label[i])

  debugged_df = pd.DataFrame(
    {'id': ids_ok,
     'qid1': q1ids_ok,
     'qid2': q2ids_ok,
     'question1': q1question_ok,
     'question2': q2question_ok,
     'is_duplicate': label_ok
    })


  debugged_df.to_csv("train_debuged.csv", sep=',', index=False)



if __name__ == '__main__':
  main()
