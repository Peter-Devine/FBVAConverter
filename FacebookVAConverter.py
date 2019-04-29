import argparse
import os
import pandas as pd
import numpy as np

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped Facebook VA dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--vad_bin_num', default="7", help='Number of bins to separate the VAD values into (if you edit this you need to also edit the BERT classifier code)')

args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
VAD_BIN_NUM = int(args.vad_bin_num)

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

fbva = pd.read_csv(INPUT_PATH + "/dataset-fb-valence-arousal-anon.csv")

fbva["V"] = fbva[['Valence1', 'Valence2']].mean(axis=1)
fbva["A"] = fbva[['Arousal1', 'Arousal2']].mean(axis=1)

fraction = 0.2

np.random.seed(seed=42)

test_indices = np.random.choice(fbva.index, size=int(round(fraction*fbva.shape[0])), replace=False)
train_indices = fbva.index.difference(test_indices)
dev_indices = np.random.choice(train_indices, size=int(round(fraction*len(train_indices))), replace=False)
train_indices = train_indices.difference(dev_indices)

fbva_train = fbva.loc[train_indices,:]
fbva_dev = fbva.loc[dev_indices,:]
fbva_test = fbva.loc[test_indices,:]

split_dataframe = {"train": fbva_train,
                  "dev": fbva_dev,
                  "test": fbva_test}

training_bins = {"V": "",
                "A": ""}
    
for dataframe_type in ["train", "dev", "test"]:
    
    dataframe = split_dataframe[dataframe_type]
    
    emotion_columns = ["V", "A"]
    
    for column in emotion_columns:
        number_of_bins = VAD_BIN_NUM
        bin_labels = [column+str(bin_label_index+1) for bin_label_index in range(number_of_bins)]
        if dataframe_type == "train":
            binned_data = pd.cut(dataframe[column], bins=number_of_bins, retbins=True)
            bins = binned_data[1]
            bins[0] = -999
            bins[len(bins)-1] = 999
            training_bins[column] = bins
        else:
            bins = training_bins[column]
        dataframe[column + "_binned"] = pd.cut(dataframe[column], bins=bins, labels=bin_labels)
     
    dataframe.reset_index(drop=True).to_csv(OUTPUT_PATH+"/"+dataframe_type+".tsv", sep='\t', encoding="utf-8")
