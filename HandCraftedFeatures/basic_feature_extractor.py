import pandas as pd
import numpy as np
import autocorrect as at
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import lcs
import pickle
import multiprocessing

def get_wordnet_pos(treebank_tag):
  """
  return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
  """
  if treebank_tag.startswith('J'):
    return wordnet.ADJ
  elif treebank_tag.startswith('V'):
    return wordnet.VERB
  elif treebank_tag.startswith('N'):
    return wordnet.NOUN
  elif treebank_tag.startswith('R'):
    return wordnet.ADV
  else:
    # As default pos in lemmatization is Noun
    return wordnet.NOUN

lemmatizer = WordNetLemmatizer()

def fix_spelling(words_spell):
  return [at.spell(word) for word in words_spell]

def fix_stemer(pos_tags):
  return [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tags]

def fix_mean_length(words):
  return np.mean([len(word) for word in words])

def calculate_similarity(words):
  return float(len(set(words[0]+words[1])))/(len(words[0]) + len(words[1]))

def calculate_lcs(words):
  return len(lcs.lcs(words[0][0:10], words[1][0:10]))

def similar_ratio(words_list):
  import pdb; pdb.set_trace()
  return pool.map(calculate_similarity, words_list)

def longest_common_seq(words_list):
  return pool.map(calculate_lcs, words_list)

DEBUGGED_FILE = "train_debuged.csv" # or test_debuged.csv
FILE_TYPE     = "train" # or test

pool = multiprocessing.Pool(600)

def main():

  if(FILE_TYPE is "train"):
    train_data = pd.read_csv(DEBUGGED_FILE)
    ids = train_data["id"].tolist()
    question1 = train_data["question1"].tolist()
    question2 = train_data["question2"].tolist()
    label = train_data["is_duplicate"].tolist()
  else if(FILE_TYPE is "test"):
    train_data = pd.read_csv(DEBUGGED_FILE)
    ids = train_data["id"].tolist()
    question1 = train_data["question1"].tolist()
    question2 = train_data["question2"].tolist()
    label = None


  questions1_correct = []
  questions1_stemmed = []
  questions1_tags = []
  questions1_tags_full = []

  print("Paralel tokens of Q1")
  words = pool.map(nltk.word_tokenize, question1)
  print("Paralel spelling of Q1")
  words_spell = pool.map(fix_spelling, words)
  print("Paralel tag of Q1")
  pos_tags = pool.map(nltk.pos_tag, words_spell)
  print("Paralel stemmer of Q1")
  words1_stemmer = pool.map(fix_stemer, pos_tags)
  questions1_tags_full = pos_tags
  questions1_tags = []
  for tag_full in questions1_tags_full:
    questions1_tags.append([tag for tk, tag in tag_full])
  for words in words_spell:
    questions1_correct.append(" ".join(words))
  for words in words1_stemmer:
    questions1_stemmed.append(" ".join(words))

  questions2_correct = []
  questions2_stemmed = []
  questions2_tags = []
  questions2_tags_full = []

  print("Paralel tokens of Q2")
  words = pool.map(nltk.word_tokenize, question2)
  print("Paralel spelling of Q2")
  words_spell = pool.map(fix_spelling, words)
  print("Paralel tag of Q2")
  pos_tags = pool.map(nltk.pos_tag, words_spell)
  print("Paralel stemmer of Q2")
  words2_stemmer = pool.map(fix_stemer, pos_tags)
  questions2_tags_full = pos_tags
  questions2_tags = []
  for tag_full in questions1_tags_full:
    questions2_tags.append([tag for tk, tag in tag_full])
  for words in words_spell:
    questions2_correct.append(" ".join(words))
  for words in words2_stemmer:
    questions2_stemmed.append(" ".join(words))

  spell_correct_text_df = pd.DataFrame(
    {'id': ids,
     'question1': questions1_correct,
     'question2': questions2_correct,
     'is_duplicate': label
    })


  spell_correct_text_df.to_csv(FILE_TYPE + "_spell.csv", sep=',', index=False)

  steammed_text_df = pd.DataFrame(
    {'id': ids,
     'question1': questions1_stemmed,
     'question2': questions2_stemmed,
     'is_duplicate': label
    })


  steammed_text_df.to_csv(FILE_TYPE + "_stemmed.csv", sep=',', index=False)

  print("Finish generating files")

  print("Saving processed words")
  pre_process_wd = {}
  pre_process_wd["questions1_stemmed"] = questions1_stemmed
  pre_process_wd["questions2_stemmed"] = questions2_stemmed
  pre_process_wd["words1_stemmer"] = words1_stemmer
  pre_process_wd["words2_stemmer"] = words2_stemmer
  pre_process_wd["questions1_tags"] = questions1_tags
  pre_process_wd["questions2_tags"] = questions2_tags
  feature_fd = open(FILE_TYPE + '_pre_processed_words.pickle', 'wb')
  pickle.dump(pre_process_wd, feature_fd, protocol=pickle.HIGHEST_PROTOCOL)


  print("Starting getting features")

  features_question = {}
  features_question["char_length_Q1"] = []
  features_question["char_length_Q2"] = []
  features_question["word_length_Q1"] = []
  features_question["word_length_Q2"] = []
  features_question["mean_word_length_Q1"] = []
  features_question["mean_word_length_Q2"] = []
  features_question["similar_words_ratio"] = []
  features_question["similar_tags_ratio"] = []
  features_question["longest_common_sequence"] = []
  features_question["longest_common_sequence_tags"] = []

  features_question["char_length_Q1"] = pool.map(len, questions1_stemmed)
  features_question["char_length_Q2"] = pool.map(len, questions2_stemmed)
  features_question["word_length_Q1"] = pool.map(len, words1_stemmer)
  features_question["word_length_Q2"] = pool.map(len, words2_stemmer)

  features_question["mean_word_length_Q1"] = pool.map(fix_mean_length, words1_stemmer)
  features_question["mean_word_length_Q2"] = pool.map(fix_mean_length, words2_stemmer)
  questions_pair = [[q1, q2] for q1, q2 in zip(words1_stemmer, words2_stemmer)]
  features_question["similar_words_ratio"] = similar_ratio(questions_pair)
  features_question["longest_common_sequence"] = longest_common_seq(questions_pair)
  questions_tag_pair = [[q1, q2] for q1, q2 in zip(questions1_tags, questions2_tags)]
  features_question["similar_tags_ratio"] = similar_ratio(questions_tag_pair)
  features_question["longest_common_sequence_tags"] = longest_common_seq(questions_tag_pair)

  feature_fd = open(FILE_TYPE + '_features_basic.pickle', 'wb')
  pickle.dump(features_question, feature_fd, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  main()
