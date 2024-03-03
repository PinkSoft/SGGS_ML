# K-fold splitting ensemble
# Deep learning model has its randomness and uncertainty. To achieve a more robust and reliable model, we adopted K-fold spliting to create an ensemble model.
# We split the training dataset into k folds. We use 1 fold as the validation dataset during training, and the remaining k-1 folds as the training dataset. This way we will have k models with each model learning different variations of the dataset.
# Then the predictions of all the k models are averaged to get the final prediction.

import sklearn
from sklearn.model_selection import KFold
import Data
from dotenv import load_dotenv
from pathlib import Path
import os


load_dotenv()
env_path = Path('.')/'.env'
#print(env_path)
load_dotenv(dotenv_path=env_path)
path = os.getenv('DATA_PATH')

print(path)

x_train, y_train = Data.load_data(path = path)
print(len(x_train), len(y_train))
#set the number of folds -- k
num_folds = 10

# Define per-fold score containers
fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []
loss_per_fold = []

kfold = KFold(n_splits=num_folds, shuffle=True, random_state = 4)

Val_precision_per_fold = []
Val_recall_per_fold = []
Val_f1_per_fold = []

Test_precision_per_fold = []
Test_recall_per_fold = []
Test_f1_per_fold = []


#split the dataset into k folds, save the index of training and validation data
split = []
for train, val in kfold.split(x_train, y_train):
  com = {'train': train, 'val': val}
  split.append(com)
  #print('train')
  #print(com['train'])
  #print('val')
  #print(com['val'])
  #print(' ')
  
  #print(x_train)
  #print(' ')
  #print(x_train)
  #print(' ')
  
x=(split[0]['train'][100])
print(x_train[x])
print(y_train[x])
  

