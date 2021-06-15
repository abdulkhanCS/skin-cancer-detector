import pandas as pd
import shutil, os
import pathlib

#Read csv file
data = pd.read_csv("ISIC_2019_Training_GroundTruth.csv") 

#Configure path to data folder
data_dir = pathlib.Path('./ISIC_2019_Training_Input')

#   -4:None -8:Squamous cell carcinoma    -12:Vascular lesion -16:Dermatofibroma  -20:Benign keratosis    -24:Actinic keratosis
#   -28:Basal cell carcinoma    -32:Melanocytic nevus   -36:Melanoma

with open('ISIC_2019_Training_GroundTruth.csv') as fp:
    for line in fp:
        if line[-8] == "1":
            source = './ISIC_2019_Training_Input/' + line[0:-37]+".jpg"
            shutil.move(source, './ISIC_2019_Training_Input/Squamous_Cell_Carcinoma')
            print(source)
