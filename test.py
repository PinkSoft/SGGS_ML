import glob
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from zipfile import ZipFile
from dotenv import load_dotenv
from sklearn.model_selection import KFold
import data



load_dotenv()
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
PATH = os.getenv("DATA_PATH_512")

print(PATH)

num_folds = 5

images, masks = data.load_data(PATH, mask_set = 'Multi')

print("Number of training samples:", len(images))
print("Number of validation samples:", len(masks))

kfold = KFold(n_splits=num_folds, shuffle=True, random_state = 4)
#print(images)

split = []
for train, valid in kfold.split(images, masks):
    #total_size = len(images)
    #split_size = int(split * total_size)
    #train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    #train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)
    #valid, test
        
    com = {'train': train, 'valid': valid}
    #print(valid)
    split.append(com)
    #print('train')
    #print(com['train'])
    print('valid')
    print(com['valid'])
    print(valid[1])
    
   
    #print(x_train)
    #print(' ')
    #print(x_train)
    #print(' ')
    #print(com)
print(split[1])   
    
    
          