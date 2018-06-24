import pandas as pd

TEST_CSV_FILE = "test.csv"

def main():
  train_data = pd.read_csv(TEST_CSV_FILE)
  train_data = train_data.dropna(axis=0, how='any')

  ids = train_data["test_id"].tolist()
  question1 = train_data["question1"].tolist()
  question2 = train_data["question2"].tolist()

  min_length_valid_question = 6;
  ids_ok = []
  q1question_ok = []
  q2question_ok = []
  for i in range(0, len(ids)):
    q1 = question1[i]
    q2 = question2[i]
    try:
      if(not isinstance(q1, basestring) or not isinstance(q2, basestring)):
        q1 = str(q1)
        q2 = str(q2)
        continue
      if(len(q1) < min_length_valid_question or len(q2) < min_length_valid_question):
        print "Removing bug question id %d" % ids[i]
        print "Q1: ", q1
        print "Q2: ", q2
        continue
    except :
      print "Exception!! - Bad question removing"
      print "Removing bug question id %d" % ids[i]
      print "Q1: ", q1
      print "Q2: ", q2
      continue
    ids_ok.append(ids[i])
    q1question_ok.append(question1[i])
    q2question_ok.append(question2[i])

  debugged_df = pd.DataFrame(
    {'id': ids_ok,
     'question1': q1question_ok,
     'question2': q2question_ok,
    })


  debugged_df.to_csv("test_debuged.csv", sep=',', index=False)



if __name__ == '__main__':
  main()
