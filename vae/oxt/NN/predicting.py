import os
import pandas as pd
import numpy as np
from keras.models import load_model
import sys

path_csv = sys.argv[1]
filename_to_save = sys.argv[2]
df_test = pd.read_csv(path_csv)
X_test = np.array([[[int(nuc == base) for nuc in 'ACGT'] for base in seq] for seq in df_test['sequence']])

models = ['ConvModel', 'DenseModel', 'LSTMModel', 'ConvLSTMModel', 'LSTMAttentionModel']
results = pd.DataFrame()
results['sequence'] = df_test['sequence']

for model_class in models:
    folder = f"./{model_class}"
    model_files = [f for f in os.listdir(folder) if f.endswith('.h5')]

    for model_name in model_files:
        model_path = os.path.join(folder, model_name)
        model = load_model(model_path)

        probabilities = model.predict(X_test)
        prob_class_1 = probabilities[:, 1] 

        results[f"{model_class}_prob_1_{model_name}"] = prob_class_1


average_probabilities = results.loc[:, results.columns != 'sequence'].mean(axis=1)
results["average_probability"] = average_probabilities
results["max_probability"] = results.loc[:, results.columns != 'sequence'].max(axis=1)
results["min_probability"] = results.loc[:, results.columns != 'sequence'].min(axis=1)


# results["max_probability_model"] = results.drop(['sequence', 'average_probability', 'max_probability', 'min_probability'], axis=1).idxmax(axis=1)
# results["min_probability_model"] = results.drop(['sequence', 'average_probability', 'max_probability', 'min_probability'], axis=1).idxmin(axis=1)


results.to_csv(filename_to_save, index=False)
