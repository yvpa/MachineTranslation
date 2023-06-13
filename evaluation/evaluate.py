import sacrebleu


def load_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
    return lines


def compute_score():
    pass


if __name__ == '__main__':
    datasets = {'raramuri-spanish': 'tar', 'shipibo_konibo-spanish': 'shp',
                'ashaninka-spanish': 'cni', 'wixarika-spanish': 'hch', 'aymara-spanish': 'aym', 'bribri-spanish': 'bzd', 'guarani-spanish': 'gn', 'hñähñu-spanish': 'oto', 'nahuatl-spanish': 'nah',
                'quechua-spanish': 'quy'}
    EPOCHS = 30

    results_chrf = {}
    results_bleu = {}
    for k, v in datasets.items():
        pred_filename = f'../neural_network/results/{k}/baseline_{EPOCHS}_all_epoch_result'
        # pred_filename = f'../results/pre_neural_all/{k}/test.translated.all.{v}'
        real_filename = f'../data/{k}/test.es'
        pred = load_file(pred_filename)
        real = load_file(real_filename)
        assert len(pred) == len(real)
        chrf = sacrebleu.corpus_chrf(pred, real)
        bleu = sacrebleu.corpus_bleu(pred, real)
        results_chrf[k] = chrf
        results_bleu[k] = bleu.format(score_only=True)

    with open(f'./neural_30_all_results', 'w+') as f:
        f.write(f'Baseline CHRF Metrics Result:\n')
        for k, v in results_chrf.items():
            f.write(f'{k}: {v} \n')
        f.write(f'Baseline BLEU Metrics Result:\n')
        for k, v in results_bleu.items():
            f.write(f'{k}: {v} \n')
