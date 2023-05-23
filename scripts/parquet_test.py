import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
parquet = pd.read_parquet('/home/jahn/pokemon-blip-captions/data/train-00000-of-00001-566cc9b19d7203f8.parquet', engine='pyarrow')
#parquet = pd.read_parquet('/home/jahn/man_dataset.parquet', engine='pyarrow')
#print(parquet.image[0]['bytes'])

table = pa.Table.from_pandas(parquet)
parquet_schema = table.schema
print(parquet_schema)
# parquet_writer = pq.ParquetWriter("test.parquet", parquet_schema, compression='snappy')
# # Write CSV chunk to the parquet file
# table = pa.Table.from_pandas(parquet)
# parquet_writer.write_table(table)
# parquet_writer.close()
