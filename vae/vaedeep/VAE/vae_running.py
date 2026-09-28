from vaemodels import VAEModel
import pandas as pd
import numpy as np
import glob
import sys
import os


list_of_file = glob.glob('../data/vae_0_*.csv')

for file_csv in list_of_file:
    file_base_name = os.path.basename(file_csv)[:-4]  # Get only the file name without the '.csv'
    df_all = pd.read_csv(file_csv)
    df = df_all[df_all['category'] == 0]
    df.reset_index(drop=True, inplace=True)
    X = np.array([[[int(nuc == base) for nuc in 'ACGT'] for base in seq] for seq in df['sequence']])
    Y = X.copy()

    latent_dim = 2

    architectures = ['dense', 'lstm', 'conv']

    all_sequences = []

    for architecture in architectures:
        input_shape = (18, 4, 1) if architecture == 'conv' else (18, 4)
        vae = VAEModel(input_shape, latent_dim, architecture=architecture)
        vae.train(X, Y, epochs=100, batch_size=2, verbose=1)

        n_samples = 100000
        sequences = vae.generate_sequences(n_samples)
        sequences['architecture'] = architecture  # Add a column to indicate the architecture
        all_sequences.append(sequences)

    all_sequences_df = pd.concat(all_sequences, ignore_index=True)

    all_sequences_df.to_csv(f'../result/all_sequences_{file_base_name}.csv', index=False)
