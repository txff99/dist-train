1.start with dist_train_lm.sh
2.change configs in config/train-macbert-lm
3.change learning rate in trainer/optim_schedule
4.ignore unmasked labels when calculating loss -> /trainer/dist_macbert_trainer 95
