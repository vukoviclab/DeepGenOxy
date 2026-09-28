from vaemodels import VAEModel
import pandas as pd
import numpy as np
import sys

path_to_file = sys.argv[1]
df = pd.read_csv(path_to_file)
X = np.array([[[int(nuc == base) for nuc in 'ACGT'] for base in seq] for seq in df['sequence']])
Y = X.copy()

latent_dim = 2

architectures = ['dense', 'lstm', 'conv']

all_sequences = []

for architecture in architectures:
    input_shape = (18, 4, 1) if architecture == 'conv' else (18, 4)
    vae = VAEModel(input_shape, latent_dim, architecture=architecture)
    vae.train(X, Y, epochs=100, batch_size=2, verbose=0)
    
    n_samples = 100000
    sequences = vae.generate_sequences(n_samples)
    sequences['architecture'] = architecture  # Add a column to indicate the architecture
    all_sequences.append(sequences)

all_sequences_df = pd.concat(all_sequences, ignore_index=True)

all_sequences_df.to_csv('all_sequences.csv', index=False)
