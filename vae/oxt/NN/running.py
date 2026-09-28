import pandas as pd
from models import ConvModel, DenseModel, LSTMModel, ConvLSTMModel, LSTMAttentionModel, ConvLSTPAMModel

df = pd.read_csv('oxt.csv')

models = [
        ConvModel, 
        DenseModel, 
        LSTMModel, 
        ConvLSTMModel, 
        LSTMAttentionModel,
        ConvLSTPAMModel
        ]

for model_class in models:
    print(f"Running model: {model_class.__name__}")
    model = model_class(df)
    model.run_model()
