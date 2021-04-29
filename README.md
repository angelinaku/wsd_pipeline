# wsd_pipeline
Pipeline for training and validating word sense disambiguation (WSD) models  
Partly based on the pipeline code from the repo https://github.com/Erlemar/pytorch_tempest  

This repository contains code and necessary configuration files for training neural nets for the WSD task.  

The pipeline can be used for:  
- training biLSTM-CRF model for all-words WSD task (token level classification);  
- training model with word2vec/glove and ELMo embeddings;
- fine-tuning BERT model for lexical sample WSD task (sequence classification).  

To start training pipeline with default train config run the coomand:  

```shell
>>> python train.py
```  

This repository is heavily based on the libraries:
- [hydra](https://hydra.cc/)  
- [pytorch-lightning](https://www.pytorchlightning.ai/)
