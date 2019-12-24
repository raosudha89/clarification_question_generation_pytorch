# Repository information

This repository contains data and code for the paper below:

<i><a href="https://www.aclweb.org/anthology/N19-1013">
Answer-based Adversarial Training for Generating Clarification Questions</a></i><br/>
Sudha Rao (Sudha.Rao@microsoft.com) and Hal Daumé III (me@hal3.name)<br/>
Proceedings of NAACL-HLT 2019

# Downloading data

* Download embeddings from https://go.umd.edu/clarification_questions_embeddings
* Download data from https://go.umd.edu/clarification_question_generation_dataset 
  Unzip the two folders inside and copy them into the repository folder

# Running models on StackExchange dataset

* To train an MLE model, run src/run_main.sh 

* To train a Max-Utility model, run src/run_RL_main.sh

* To train a GAN-Utility model, run src/run_GAN_main.sh

# Running models on Amazon (Home & Kitchen) dataset

* To train an MLE model, run src/run_main_HK.sh

* To train a Max-Utility model, run src/run_RL_main_HK.sh

* To train a GAN-Utility model, run src/run_GAN_main_HK.sh

# Evaluating generated outputs

* For StackExchange dataset, reference for a subset of the test set was collected using human annotators.
  Hence we first create a version of the predictions file for which we have references by running following:
  src/evaluation/run_create_preds_for_refs.sh

* For Amazon dataset, we have references for all instances in the test set.

* We remove <UNK> tokens from the generated outputs by simply removing them from the predictions file.

* For BLEU score, run src/evaluation/run_bleu.sh

* For METEOR score, run src/evaluation/run_meteor.sh 

* For Diversity score, run src/evaluation/calculate_diversiy.sh <predictions_file>
