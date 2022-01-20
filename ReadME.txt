This code can be run by editing the .json configuration file to the desired parameters and running the following command:

python train.py -config_file config_sample.json

It will produce a model (.h5) file, a config.json file containing the training conditions of the model, and a tensorboard summary.
Once trained you can visualize the results of the saliency maps using the command replacing the .json path with your own:


python evaluate_saliency_methods.py -config_file /home/eric/PycharmProjects/didactic_sanity_check/model_saves/experiment_1/model1/2022-01-19-19:05:56/config.json

