import os
import zipfile

import pandas as pd

from data_generator import df_from_dirlist


def zip_data(df, source):
    for i in range((len(df) // 10000) + 1):
        with zipfile.ZipFile(source + f'archive{i}.zip', mode='w') as archive:
            for j, row in df[i*10000:(i+1)*10000].iterrows():
                archive.write(rf"{row['filepath']}")


def fix_image_dir(dir):
    from PIL import Image

    file_list = []

    for path, directories, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.join(path, file))

    def verify_jpeg_image(file_path):
        try:
            img = Image.open(file_path)
            img0 = img.getdata()[0]
            img.save(file_path)
            return bool(img0) and True
        except OSError:
            return False

    bads = 0
    goods = 0

    for path in file_list:
        if verify_jpeg_image(path):
            goods += 1
        else:
            bads += 1
            print(path)

    print('bads:', bads, 'goods', goods)


def unzip_data(source, destination, fix=False):
    for i in range(44):
        try:
            with zipfile.ZipFile(source + f'archive{i}.zip', mode='r') as archive:
                archive.extractall(f'{destination}/shard_{i}')
                if fix:
                    fix_image_dir(f'{destination}/shard_{i}')
        except:
            break


if __name__ == "__main__":
    """    print('getting dsets...')
    sites = ['/xcitelab/backup/dot/data/Skyline_1800',
             '/xcitelab/backup/dot/data/Skyline_1803',
             '/xcitelab/backup/dot/data/Skyline_2119',
             '/xcitelab/backup/dot/data/Skyline_2137',
             '/xcitelab/backup/dot/data/Skyline_5563',
             '/xcitelab/backup/dot/data/Skyline_6117',
             '/xcitelab/backup/dot/data/Skyline_8297',
             '/xcitelab/backup/dot/data/Skyline_1806',
             '/xcitelab/backup/dot/data/Skyline_5561',
             '/xcitelab/backup/dot/data/Skyline_5562'
             ]

    file_list = []

    for site in sites:
        print(site)
        for path, directories, files in os.walk(site):
            for file in files:
                file_list.append(os.path.join(path, file))

    df = pd.DataFrame(file_list, columns=['filepath'])
    """
    unzip_data('./', './unlabeled/', fix=True)


