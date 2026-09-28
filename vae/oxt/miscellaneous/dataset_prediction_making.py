import pandas as pd

df_oxt = pd.read_csv('oxt.csv')
df_selec = pd.read_csv('selec.csv')
df_vae = pd.read_csv('vae.csv')

oxt_sequences = df_oxt['sequence']

df_selec = df_selec[~df_selec['sequence'].isin(oxt_sequences)]
df_vae = df_vae[~df_vae['sequence'].isin(oxt_sequences)]

df_selec.to_csv('selec_filtered.csv', index=False)
df_vae.to_csv('vae_filtered.csv', index=False)

common_sequences = pd.merge(df_vae, df_selec, on='sequence')

common_sequences.to_csv('common_sequences_vae_selec.csv', index=False)
