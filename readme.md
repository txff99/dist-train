start with dist_train_lm.sh
change configs in config/train-macbert-lm
change learning rate in trainer/optim_schedule
ignore unmasked labels when calculating loss -> /trainer/dist_macbert_trainer 95