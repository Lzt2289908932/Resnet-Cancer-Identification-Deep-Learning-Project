# Resnet-Cancer-Identification-Deep-Learning-Project \n
Dataset from:https://www.kaggle.com/datasets/mani11111111111/cancer-detection-dataset\n
This Project used Resnet18 to train a model to predict cancer cell.\n
Activation：ReLU \n
Cerition：CELoss \n
Optimizer：Adam\n
I kept the freeze training code inside train.py, when I was train this model with freezing, this model early stopped at 25 epochs with acc 86%.\n
So I give up this method, and train this model again with all layers.\n
As the final result, my best model was created at 36 epochs with acc 93.12%, this model early stopped at 51 epochs.\n
After evaulate, acc reached 93.25, which shows that this model is NOT overfitting.\n
