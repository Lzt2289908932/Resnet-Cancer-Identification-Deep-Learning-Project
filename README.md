# Resnet-Cancer-Identification-Deep-Learning-Project   
Dataset from:https://www.kaggle.com/datasets/mani11111111111/cancer-detection-dataset  
This Project used Resnet18 to train a model to predict cancer cell.  
Activation：ReLU   
Cerition：CELoss   
Optimizer：Adam  
I kept the freeze training code inside train.py, when I was train this model with freezing, this model early stopped at 25 epochs with acc 86%. Which shows a bad wish.   
So I give up this method but the original code still in train.py I just simply annotate‌ it, and train this model again with all layers.  
As the final result, my best model was created at 36 epochs with acc 93.12%, this model early stopped at 51 epochs.  
After evaulate, acc reached 93.25, which shows that this model is NOT overfitting.  
To check the test result draft please find the png files in main branch
