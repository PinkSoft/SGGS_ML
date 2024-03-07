# K-fold splitting ensemble
# Deep learning model has its randomness and uncertainty. To achieve a more robust and reliable model, 
# we adopted K-fold spliting to create an ensemble model.
# We split the training dataset into k folds. We use 1 fold as the validation dataset during training, 
# and the remaining k-1 folds as the training dataset. This way we will have k models with each model 
# learning different variations of the dataset.
# Then the predictions of all the k models are averaged to get the final prediction.

import sklearn
from sklearn.model_selection import KFold
import Data

def create_k_folds(num_folds=10, path):
    x_train, y_train = Data.load_data(path = path)
    print(len(x_train), len(y_train))
    #set the number of folds -- k
    #num_folds = 10

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
        total_size = len(images)
        split_size = int(split * total_size)
        train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
        train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)
        valid, test
        
        com = {'train': train, 'valid': valid, test: 'test'}
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
    return split
      
  #x=(split[0]['train'][100])
  #print(x_train[x])
  #print(y_train[x])