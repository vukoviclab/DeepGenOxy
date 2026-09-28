import os
import pandas as pd
import numpy as np
from keras.models import load_model
import sys
import glob

for class_prob in [0,1]:
    for lig_name in ['ser','ox']:
        if 'ser' in lig_name:
            path_csv = glob.glob(f'/data1/payam/ml/vaedeep/result/*vae_{class_prob}_ser*.csv')[0]
            filename_to_save = f'/data1/payam/ml/vaedeep/result/ser_{class_prob}.csv'
        else:
            path_csv = glob.glob(f'/data1/payam/ml/vaedeep/result/*_vae_{class_prob}_oxt*.csv')[0]
            filename_to_save = f'/data1/payam/ml/vaedeep/result/oxt_{class_prob}.csv'
        df_test = pd.read_csv(path_csv)
        X_test = np.array([[[int(nuc == base) for nuc in 'ACGT'] for base in seq] for seq in df_test['sequence']])

        models = ['ConvModel', 'DenseModel', 'LSTMModel', 'ConvLSTMModel', 'LSTMAttentionModel']
        results = pd.DataFrame()
        results['sequence'] = df_test['sequence']

        for model_class in models:
            folder = f"./{model_class}"
            model_files = [f for f in os.listdir(folder) if f.endswith('.h5') and lig_name in f]

            for model_name in model_files:
                model_path = os.path.join(folder, model_name)
                model = load_model(model_path)

                probabilities = model.predict(X_test)
                prob_class_01 = probabilities[:, class_prob] 

                results[f"{model_class}_prob_{class_prob}_{model_name}"] = prob_class_01


        average_probabilities = results.loc[:, results.columns != 'sequence'].mean(axis=1)
        results["average_probability"] = average_probabilities
        results["max_probability"] = results.loc[:, results.columns != 'sequence'].max(axis=1)
        results["min_probability"] = results.loc[:, results.columns != 'sequence'].min(axis=1)




        results.to_csv(filename_to_save, index=False)
