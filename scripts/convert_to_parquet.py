import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import os
import numpy as np

csv_file = '/home/jahn/man_dataset.tsv'
parquet_file = '/home/jahn/man_dataset.parquet'
chunksize = 100_000

csv_stream = pd.read_csv(csv_file, sep='\t', chunksize=chunksize, low_memory=False)

for i, chunk in enumerate(csv_stream):
    print("Chunk", i, len(chunk))
    if i == 0:
        new_data_frame = pd.DataFrame(data=[], index=["image", "text", "class"]).T
        # new_parquet_schema = pa.Table.from_pandas(df=new_data_frame).schema

        new_parquet_schema = pa.schema([
            ('image', pa.struct([
                ('bytes', pa.binary()),
                ('path', pa.null())
            ])),
            ('text', pa.string()),
            ('class', pa.string())
        ])
        parquet_writer = pq.ParquetWriter(parquet_file, new_parquet_schema, compression='snappy')
#     # Write CSV chunk to the parquet file
#     table = pa.Table.from_pandas(chunk, schema=parquet_schema)
#     parquet_writer.write_table(table)
    loaded_images = []
    classes = []
    for item_index, csv_path in enumerate(chunk.Name):
        csv_path = os.path.basename(csv_path).replace(".jpeg", ".jpg")
        csv_extension = chunk.Extension[item_index]
        added_to_loaded_images = False

        if csv_extension == ".jpeg":
            csv_extension = ".jpg"

        for image_path in Path('/home/jahn/dir_dataset').rglob('*'):
            image_path_basename = os.path.basename(image_path).replace(".jpeg", ".jpg")
            image_path_suffix = image_path.suffix

            if image_path_suffix == ".jpeg":
                image_path_suffix = ".jpg"

            is_corresponding_path = csv_path.find(image_path_basename) != -1

            #print(os.path.isfile(image_path), image_path_suffix == csv_extension, is_corresponding_path)
            if os.path.isfile(image_path) and image_path_suffix == csv_extension and is_corresponding_path:
                with open(image_path, "rb") as image:
                    image_content = image.read()
                    image_bytes = bytes(image_content)
                    image_dict = {
                        "bytes": image_bytes,
                        "path": None
                    }

                    loaded_images.append(image_dict)

                if csv_path.find("topview") != -1:
                    classes.append("overhead")
                elif csv_path.find("bottomview") != -1:
                    classes.append("bottom")
                elif csv_path.find("leftview") != -1:
                    classes.append("left")
                elif csv_path.find("rightview") != -1:
                    classes.append("right")
                elif csv_path.find("frontview") != -1:
                    classes.append("front")
                elif csv_path.find("backview") != -1:
                    classes.append("back")
                else:
                    print("class must be top, bottom, left, right, front, or back")
                    raise ValueError

                added_to_loaded_images = True
                break
        if not added_to_loaded_images:
            print(f"찾지 못한 이미지 명: {csv_path}")

    texts = chunk.text.to_list()
    
    if len(texts) != len(loaded_images) or len(loaded_images) != len(classes):
        print(f"texts{len(texts)}, loaded images{len(loaded_images)}, and classes{len(classes)} count is different")
        raise ValueError
    
    new_data_frame = pd.DataFrame(data=[loaded_images, texts, classes], index=["image", "text", "class"]).T
    # new_parquet_schema = pa.Table.from_pandas(df=new_data_frame).schema
    # print(new_parquet_schema)
    table = pa.Table.from_pandas(new_data_frame)
    parquet_writer.write_table(table)

parquet_writer.close()

# parquet_writer.close()
