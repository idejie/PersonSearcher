from loguru import logger


class Config(object):
    def __init__(self):
        self.action = 'train'  # process, train or test
        logger.add('logs/{time:YYYY-MM-DD_HH-mm-ss}.log')
        self.logger = logger
        self.gpu_id = 0
        self.num_workers = 10
        self.vocab_dir = 'vocab'
        self.data_dir = 'data'
        self.raw_data = 'reid_raw.json'
        self.word_count_threshold = 2
        self.positive_samples = 1
        self.negative_samples = 3
        self.n_original_captions = 2
        self.max_length = 50
        self.epochs = 200
        self.amp = True
        self.embedding_size = 512
        self.images_dir = 'data/CUHK-PEDES/imgs'
        self.checkpoints_dir = 'checkpoints'
        self.rnn_hidden_size = 512
        self.output_size = 512
        self.batch_size = 32
        self.top_k = [1, 5, 10]
        self.image_size = (256, 256)
        self.eval_interval = 400
        self.test_interval = 1
        self.rnn_layers = 1
        self.rnn_dropout = 0.1
        self.backend = 'cudnn'  # nn|cudnn
        self.parallel = False
        self.web_debug = True
        self.language_lr = 1e-6
        self.cnn_lr = 1e-10
        self.step1 = 50
        self.classifier_lr = 1e-3
        self.margin = 0.2
