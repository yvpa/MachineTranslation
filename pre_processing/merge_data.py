def load_from_file(file_path):
    text = None
    with open(file_path) as f:
        text = f.read().split('\n')
    return text


def save_to_file(file_path, data):
    with open(file_path, 'a+') as f:
        for line in data:
            out = line.replace('[start]', '').replace('[end]', '')
            f.write(out+'\n')


def build_dataset(datasets, source_path, target_path, filename):
    for k, v in datasets.items():
        target_train_path = f'../data/{k}/{filename}.es'
        source_train_path = f'../data/{k}/{filename}.{v}'
        tmp_source = load_from_file(source_train_path)
        tmp_target = load_from_file(target_train_path)
        print(f'{filename}:{k}')
        assert (len(tmp_source) == len(tmp_target))

        save_to_file(source_path, tmp_target)
        save_to_file(target_path, tmp_target)


if __name__ == '__main__':
    datasets = {'raramuri-spanish': 'tar', 'shipibo_konibo-spanish': 'shp',
                'ashaninka-spanish': 'cni', 'wixarika-spanish': 'hch', 'aymara-spanish': 'aym', 'bribri-spanish': 'bzd', 'guarani-spanish': 'gn', 'hñähñu-spanish': 'oto', 'nahuatl-spanish': 'nah',
                'quechua-spanish': 'quy'}
    # build_dataset(datasets, '../data/all-spanish/test.all',
    #              '../data/all-spanish/test.es', 'test')
    # build_dataset(datasets, '../data/all-spanish/train.all',
    #              '../data/all-spanish/train.es', 'train')
    build_dataset(datasets, '../data/all-spanish/dev.all',
                  '../data/all-spanish/dev.es', 'dev')
