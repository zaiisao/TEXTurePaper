import pandas as pd
#parquet = pd.read_parquet('/home/jahn/pokemon-blip-captions/data/train-00000-of-00001-566cc9b19d7203f8.parquet', engine='pyarrow')
parquet = pd.read_parquet('/home/jahn/man_dataset.parquet', engine='pyarrow')
print(parquet)
