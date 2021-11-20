# Counterfactual_Fairness_Education

The code in this repository belongs to a study investigating the Dutch selection procedure from primary to secondary school, which appears to have undesirable bias.

The fairness metrics _fairness through unawareness_, _equality of opportunity_ and, in particular, _counterfactual fairness_ are tested on the problem. The individual level counterfactual fairness metric states that an algorithm is "fair" if the outcome remains the same in a counterfactual world created by altering the value of a sensitive attribute, such as ethnicity.

In order to do this, the _causality_ framework was used for constructing a causal graph on the problem and _approximate inference_, a _causal effect variational autoencoder_ (CEVAE) was used to estimate the relations between the relevant variables.

The data is only available in a secure environment of Centraal Bureau voor de Statistiek (CBS) for privacy reasons.

## clean_data
The clean_data directory contains files for cleaning the data and applying an impution technique for part of the missing data.

## causal_network
The causal_network directory contains files for constructing the CEVAE model and fair machine learning models to predict the educational level of pupils.

## Usage
Subdirectory clean_data:

``main.py``

Subdirectory causal_network:
s
``main.py [-lr LR] [-hDim HDIM] [-uDim UDIM] [-rep REP] [-nIter_CEVAE NITER_CEVAE] [-nIter NITER] [-batchSize BATCHSIZE] [-nSamplesU NSAMPLESU] [-evalIter EVALITER] [-dataset DATASET] [-device DEVICE] [-test_size TEST_SIZE] [-filename FILENAME] [-model_name MODEL_NAME]``
