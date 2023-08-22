import csv
import wandb
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, Features
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import AutoModelForSequenceClassification
from datasets import load_metric
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
import argparse
import csv
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def tokenize_blind_data(example):
  return tokenizer(example['bio'], padding='max_length', truncation = True)


def tokenize_data(example, tokenizer):
  return tokenizer(example['bio'], padding='max_length', truncation = True)

def transform_labels(example):
  label = example['label']
  if label == '[REAL]':
    num = 0
  
  else:
    num = 1

  return {'labels': num}

def getResultsOnBlindSet(model):
  dataset = load_dataset('csv',\
                         data_files={'blind_test': './processed_blind.test.txt'}, delimiter = '\n')


  dataset = dataset.map(tokenize_blind_data, batched=True)

  test_args = TrainingArguments(
      output_dir = './bert_base_finetuned_fake_bio_detector/checkpoint-1800',
      do_train = False,
      do_predict = True,
      per_device_eval_batch_size = 64,   
      dataloader_drop_last = False    
  )

  # init trainer
  trainer = Trainer(
                model = model, 
                args = test_args, 
                compute_metrics = compute_metrics)

  test_results = trainer.predict(dataset['blind_test'].select(range(2)))
  test_labels = test_results.predictions.argmax(-1)
  return test_labels


def getResultsOnTestSet(model):
  # code for generating results on testset
  model = AutoModelForSequenceClassification.from_pretrained('./bert_base_finetuned_fake_bio_detector/checkpoint-1800',  num_labels=2)
  # test on test set

  dataset = load_dataset('csv',\
                         data_files={'test': './combined_test.txt'}, delimiter = '\t')

  dataset = dataset.map(lambda k: tokenize_data(k, tokenizer), batched=True)

  dataset = dataset.map(transform_labels, remove_columns = ['bio', 'label'])

  dataset = dataset.map(tokenize_blind_data, batched=True)

  test_args = TrainingArguments(
      output_dir = './bert_base_finetuned_fake_bio_detector/checkpoint-1800',
      do_train = False,
      do_predict = True,
      per_device_eval_batch_size = 64,   
      dataloader_drop_last = False    
  )

  # init trainer
  trainer = Trainer(
                model = model, 
                args = test_args, 
                compute_metrics = compute_metrics)

  preds = predictOnTestSet(trainer, dataset)

  return preds	

"""## Write labels to CSV"""

def write__results_to_csv():

  with open('./blind_test_results.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames = {'label'})
    writer.writeheader()
    for l in test_labels:
      if l == 1:
        writer.writerow({'label' : '[FAKE]'})
      else:
        writer.writerow({'label' : '[REAL]'})


def compute_metrics(eval_pred):
  acc_metric = load_metric('accuracy')
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)
  return acc_metric.compute(predictions=predictions, references=labels)


def plot_confusion_matrix(test_result):
  preds = test_result.predictions.argmax(-1)
  cm = confusion_matrix(preds, test_result.label_ids)
  plt.figure(figsize=(6, 6))
  sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['REAL', 'FAKE'],
              yticklabels=['REAL', 'FAKE'])
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.show()

  print(classification_report(test_result.label_ids, preds, target_names=['Predicted Fake', 'Predicted True']))

def main():
  ## code for generating labels on blind set
  model = AutoModelForSequenceClassification.from_pretrained('./bert_base_finetuned_fake_bio_detector/checkpoint-1800',  num_labels=2)
  test_labels = getResultsOnBlindSet(model)
  write__results_to_csv()

  # get results on test set and plot confusion matrix
  preds = getResultsOnTestSet(model)
  plot_confusion_matrix(preds)



 if __name__ == "__main__":
  main()