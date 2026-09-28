import pandas as pd
import glob
from models import ConvModel, DenseModel, LSTMModel, ConvLSTMModel, LSTMAttentionModel, ConvLSTPAMModel

csv_files = glob.glob('*.csv')

models = [
        ConvModel, 
        DenseModel, 
        LSTMModel, 
        ConvLSTMModel, 
        LSTMAttentionModel,
        ConvLSTPAMModel
        ]

for csv_file in csv_files:
    for model_class in models:
        print(f"Running model: {model_class.__name__}")
        model = model_class(csv_file)
        model.run_model()
