import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.backends import cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import CUHK_PEDES
from model import GNA_RNN, RankLoss
from utils.config import Config
from utils.preprocess import *

config = Config()
if config.action in ['train', 'test', 'web']:
    if config.amp:
        from torch.cuda.amp import autocast
    if config.parallel:
        from torch import distributed

        # init distributed backend
        distributed.init_process_group(backend='nccl')
        # configuration for different GPUs


class Model(object):
    """the class of model

    Attributes:

    """

    def __init__(self, conf):
        conf.logger.info(f'CUDA is available? {torch.cuda.is_available()}')
        if type(conf.gpu_id) == int and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{conf.gpu_id}')

        elif type(conf.gpu_id) == list and len(conf.gpu_id) > 0 and torch.cuda.device_count() > 1:
            local_rank = distributed.get_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cpu')
            self.device = device
        torch.cuda.set_device(self.device)
        conf.logger.info(self.device)
        if conf.backend == 'cudnn' and torch.cuda.is_available():
            cudnn.benchmark = True

        self.logger = conf.logger
        self.vocab_dir = conf.vocab_dir
        self.data_dir = conf.data_dir
        self.raw_data = conf.raw_data
        self.word_count_threshold = conf.word_count_threshold
        self.conf = conf
        # load data
        if conf.action != 'process':
            train_set, valid_set, test_set, vocab = self.load_data()
            conf.vocab_size = vocab['UNK'] + 1
            self.train_set = CUHK_PEDES(conf, train_set, image_caption='caption')
            self.valid_img_set = CUHK_PEDES(conf, valid_set, image_caption='image')
            self.valid_cap_set = CUHK_PEDES(conf, valid_set, image_caption='caption')
            self.test_img_set = CUHK_PEDES(conf, test_set, image_caption='image')
            self.test_cap_set = CUHK_PEDES(conf, test_set, image_caption='caption')
            self.conf.num_classes = int(
                self.train_set.num_classes + self.valid_img_set.num_classes + self.test_img_set.num_classes)
            self.logger.info(f'all num_classes: {self.conf.num_classes}')
            if conf.parallel:
                self.train_loader = DataLoader(self.train_set, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                               shuffle=True,
                                               sampler=DistributedSampler(self.train_set))
                self.valid_img_loader = DataLoader(self.valid_img_set, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   sampler=DistributedSampler(self.valid_img_set))
                self.valid_cap_loader = DataLoader(self.valid_cap_set, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   sampler=DistributedSampler(self.valid_cap_set))
                self.test_img_loader = DataLoader(self.test_img_set, batch_size=conf.batch_size,
                                                  num_workers=conf.num_workers,
                                                  sampler=DistributedSampler(self.test_img_set))
                self.test_cap_loader = DataLoader(self.test_cap_set, batch_size=conf.batch_size,
                                                  num_workers=conf.num_workers,
                                                  sampler=DistributedSampler(self.test_cap_set))
            else:
                self.train_loader = DataLoader(self.train_set, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                               shuffle=True)
                self.valid_img_loader = DataLoader(self.valid_img_set, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers)
                self.valid_cap_loader = DataLoader(self.valid_cap_set, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers)
                self.test_img_loader = DataLoader(self.test_img_set, batch_size=conf.batch_size,
                                                  num_workers=conf.num_workers)
                self.test_cap_loader = DataLoader(self.test_cap_set, batch_size=conf.batch_size,
                                                  num_workers=conf.num_workers)

            # init network
            self.net = GNA_RNN(conf)
            self.net.cuda()
            if conf.parallel:
                self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[local_rank],
                                                                     output_device=local_rank)
                self.optimizer = Adam([
                    {'params': self.net.module.language_subnet.parameters(), 'lr': conf.language_lr},
                    {'params': self.net.module.cnn.parameters(), 'lr': conf.cnn_lr},
                    {'params': self.net.module.img_classifier.parameters(), 'lr': conf.classifier_lr},
                    {'params': self.net.module.cap_classifier.parameters(), 'lr': conf.classifier_lr}
                ])
            else:
                self.optimizer = Adam([
                    {'params': self.net.language_subnet.parameters(), 'lr': conf.language_lr},
                    {'params': self.net.cnn.parameters(), 'lr': conf.cnn_lr},
                    {'params': self.net.img_classifier.parameters(), 'lr': conf.classifier_lr},
                    {'params': self.net.cap_classifier.parameters(), 'lr': conf.classifier_lr}
                ])

            self.classifier_loss = nn.CrossEntropyLoss()
            self.rank_loss = RankLoss(conf)
            all_steps = self.conf.epochs * len(self.train_loader)
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                                  T_max=all_steps / len(local_rank) if conf.parallel else all_steps)
            self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def load_data(self):
        train_set_path = os.path.join(self.data_dir, 'train_set.json')
        with open(train_set_path, 'r', encoding='utf8') as f:
            train_set = json.load(f)
        valid_set_path = os.path.join(self.data_dir, 'valid_set.json')
        with open(valid_set_path, 'r', encoding='utf8') as f:
            valid_set = json.load(f)
        test_set_path = os.path.join(self.data_dir, 'test_set.json')
        with open(test_set_path, 'r', encoding='utf8') as f:
            test_set = json.load(f)
        w2i_path = os.path.join(self.vocab_dir, 'w2i.json')
        with open(w2i_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        return train_set, valid_set, test_set, vocab

    def process(self):
        # load images list
        raw_data_path = os.path.join(self.data_dir, self.raw_data)
        with open(raw_data_path, 'r', encoding='utf8') as f:
            images_info = json.load(f)

        #  tokenize captions
        # images_tokenized = tokenize(images)

        # create the vocab
        vocab, images_info = build_vocab(images_info, self.word_count_threshold)
        i2w = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
        w2i = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

        # save vocab-index map
        if not os.path.exists(self.vocab_dir):
            os.mkdir(self.vocab_dir)
        i2w_path = os.path.join(self.vocab_dir, 'i2w.json')
        with open(i2w_path, 'w', encoding='utf8') as f:
            json.dump(i2w, f, indent=2)
        w2i_path = os.path.join(self.vocab_dir, 'w2i.json')
        with open(w2i_path, 'w', encoding='utf8') as f:
            json.dump(w2i, f, indent=2)

        # encode captions
        train_set, val_set, test_set = encode_captions(images_info, w2i)

        # save the splitting dataset
        train_set_path = os.path.join(self.data_dir, 'train_set.json')
        with open(train_set_path, 'w', encoding='utf8') as f:
            json.dump(train_set, f)
        valid_set_path = os.path.join(self.data_dir, 'valid_set.json')
        with open(valid_set_path, 'w', encoding='utf8') as f:
            json.dump(val_set, f)
        test_set_path = os.path.join(self.data_dir, 'test_set.json')
        with open(test_set_path, 'w', encoding='utf8') as f:
            json.dump(test_set, f)

    def train(self):
        # train stage
        if self.conf.amp:
            scaler = torch.cuda.amp.GradScaler()
        for e in range(self.conf.epochs):
            for b, (index, images, captions, img_ids, p_ids) in enumerate(self.train_loader):
                self.net.train()
                self.optimizer.zero_grad()
                if self.conf.gpu_id != -1:
                    self.net.cuda()
                    images = images.cuda()
                    captions = captions.cuda()
                    img_ids = img_ids.cuda()
                    # p_ids = p_ids.cuda()
                if self.conf.amp:
                    with autocast():
                        img_feats, img_out, cap_feats, cap_out = self.net(images, captions)

                        loss1 = self.classifier_loss(img_out, img_ids) + self.classifier_loss(cap_out, img_ids)
                        loss2 = self.rank_loss(img_feats, cap_feats, img_ids, img_ids)
                        if e >= self.conf.step1:
                            loss = loss1 + loss2
                        else:
                            loss = loss1
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    img_feats, img_out, cap_feats, cap_out = self.net(images, captions)
                    loss1 = self.classifier_loss(img_out, img_ids) + self.classifier_loss(cap_out, img_ids)
                    loss2 = self.rank_loss(img_feats, cap_feats, img_ids, img_ids)
                    if e >= self.conf.step1:
                        loss = loss1 + loss2
                    else:
                        loss = loss1

                    loss.backward()
                    self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()

                self.logger.info(
                    f'Epoch {e}/{self.conf.epochs} Batch {b}/{len(self.train_loader)}, loss_classifier:{loss1.item():.4f},loss_rank:{loss2.item():.4f}')
                if (b + 1) % self.conf.eval_interval == 0:
                    self.eval()

            self.save_checkpoint(e)
            if (e + 1) % self.conf.test_interval == 0:
                self.test()

    def test(self):
        # test stage
        self.net.eval()
        n_query = len(self.test_cap_set)
        n_database = len(self.test_cap_set)
        out_matrix = np.zeros((n_query, n_database))
        labels_matrix = np.zeros((n_query, n_database))
        q_pred = []
        q_true = []
        d_pred = []
        d_true = []
        with torch.no_grad():
            # images: for d in db
            query_cap_vectors = []
            db_img_vectors = []
            for q, (indexes, _, captions, img_q_ids, p_q_ids) in enumerate(self.test_cap_loader):
                if self.conf.gpu_id != -1:
                    captions = captions.cuda()
                    # repeat image for batch input
                    # p_d_id_repeat = p_d_id.repeat(len(captions), 1)
                if self.conf.amp:
                    with autocast():
                        _, _, cap_q_feats, cap_q_out = self.net(None, captions)
                else:
                    _, _, cap_q_feats, cap_q_out = self.net(None, captions)
                query_cap_vectors.append((indexes, img_q_ids, cap_q_feats))
                cap_q_out = nn.Softmax(dim=1)(cap_q_out)
                y_pred = cap_q_out.argmax(dim=1)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = img_q_ids.cpu().detach().numpy()
                q_pred.append(y_pred)
                q_true.append(y_true)
            q_pred = np.concatenate(q_pred)
            q_true = np.concatenate(q_true)
            self.logger.info('test caption classifier:')
            self.classifier_report(q_true, q_pred)
            for d, (indexes, images, _, img_d_ids, p_d_ids) in enumerate(self.test_img_loader):
                if self.conf.gpu_id != -1:
                    images = images.cuda()
                if self.conf.amp:
                    with autocast():
                        img_d_feats, img_d_out, _, _ = self.net(images)
                else:
                    img_d_feats, img_d_out, _, _ = self.net(images)
                db_img_vectors.append((indexes, img_d_ids, img_d_feats))
                img_d_out = nn.Softmax(dim=1)(img_d_out)
                y_pred = img_d_out.argmax(dim=1)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = img_d_ids.cpu().detach().numpy()
                d_pred.append(y_pred)
                d_true.append(y_true)
            d_pred = np.concatenate(d_pred)
            d_true = np.concatenate(d_true)
            self.logger.info('test image classifier:')
            self.classifier_report(d_true, d_pred)
            for item1 in query_cap_vectors:
                indexes_q, cap_q_ids, q_vector = item1
                for item2 in db_img_vectors:
                    indexes_d, img_d_ids, d_vector = item2
                    sim = self.cos(q_vector.unsqueeze(1), d_vector)
                    out_matrix[indexes_q, :][:, indexes_d] = sim.cpu().detach().numpy()
                    labels = (cap_q_ids.repeat(len(indexes_d), 1).t() == img_d_ids.repeat(len(indexes_q), 1)) + 0
                    labels_matrix[indexes_q, :][:, indexes_d] = labels.cpu().detach().numpy()

        out_matrix = torch.from_numpy(out_matrix)
        labels_matrix = torch.from_numpy(labels_matrix)
        metrics = self.calculate_metrics(out_matrix, labels_matrix)
        return metrics

    def classifier_report(self, y_true, y_pred):
        self.logger.info(accuracy_score(y_true, y_pred))

    def eval(self):
        # eval stage
        self.net.eval()
        n_query = len(self.valid_cap_set)
        n_database = len(self.valid_img_set)
        out_matrix = np.zeros((n_query, n_database))
        labels_matrix = np.zeros((n_query, n_database))
        q_pred = []
        q_true = []
        d_pred = []
        d_true = []
        with torch.no_grad():
            # images: for d in db
            query_cap_vectors = []
            db_img_vectors = []
            for q, (indexes, _, captions, img_q_ids, p_q_ids) in enumerate(self.valid_cap_loader):
                if self.conf.gpu_id != -1:
                    captions = captions.cuda()
                    # repeat image for batch input
                    # p_d_id_repeat = p_d_id.repeat(len(captions), 1)
                if self.conf.amp:
                    with autocast():
                        _, _, cap_q_feats, cap_q_out = self.net(None, captions)
                else:
                    _, _, cap_q_feats, cap_q_out = self.net(None, captions)
                query_cap_vectors.append((indexes, img_q_ids, cap_q_feats))
                cap_q_out = nn.Softmax(dim=1)(cap_q_out)
                y_pred = cap_q_out.argmax(dim=1)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = img_q_ids.cpu().detach().numpy()
                q_pred.append(y_pred)
                q_true.append(y_true)
            q_pred = np.concatenate(q_pred)
            q_true = np.concatenate(q_true)
            self.logger.info('caption classifier:')
            self.classifier_report(q_true, q_pred)
            for d, (indexes, images, _, img_d_ids, p_d_ids) in enumerate(self.valid_img_loader):
                if self.conf.gpu_id != -1:
                    images = images.cuda()
                if self.conf.amp:
                    with autocast():
                        img_d_feats, img_d_out, _, _ = self.net(images)
                else:
                    img_d_feats, img_d_out, _, _ = self.net(images)
                db_img_vectors.append((indexes, img_d_ids, img_d_feats))
                img_d_out = nn.Softmax(dim=1)(img_d_out)
                y_pred = img_d_out.argmax(dim=1)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = img_d_ids.cpu().detach().numpy()
                d_pred.append(y_pred)
                d_true.append(y_true)
            d_pred = np.concatenate(d_pred)
            d_true = np.concatenate(d_true)
            self.logger.info('image classifier:')
            self.classifier_report(d_true, d_pred)
            for item1 in query_cap_vectors:
                indexes_q, cap_q_ids, q_vector = item1
                for item2 in db_img_vectors:
                    indexes_d, img_d_ids, d_vector = item2
                    sim = self.cos(q_vector.unsqueeze(1), d_vector)
                    out_matrix[indexes_q, :][:, indexes_d] = sim.cpu().detach().numpy()
                    labels = (cap_q_ids.repeat(len(indexes_d), 1).t() == img_d_ids.repeat(len(indexes_q), 1)) + 0
                    labels_matrix[indexes_q, :][:, indexes_d] = labels.cpu().detach().numpy()

        out_matrix = torch.from_numpy(out_matrix)
        labels_matrix = torch.from_numpy(labels_matrix)
        metrics = self.calculate_metrics(out_matrix, labels_matrix)
        return metrics

    def query_transform(self, query):
        w2i_file = os.path.join(self.vocab_dir, 'w2i.json')
        with open(w2i_file, 'r', encoding='utf8') as f:
            w2i = json.load(f)
        tokens = ''.join(c for c in query.strip().lower() if c not in string.punctuation)
        tokens = tokens.split()
        query_vector = []
        for token in tokens:
            query_vector.append(w2i.get(token, w2i['UNK']))
        caption = np.zeros(self.conf.max_length)
        for i, cap_i in enumerate(query_vector):
            if i < self.conf.max_length:
                caption[i] = cap_i
            else:
                break
        return caption

    def web(self):
        from flask import Flask, request
        from flask import render_template
        app = Flask(__name__, template_folder='web', static_folder='data/CUHK-PEDES/imgs/', static_url_path='')

        self.load_checkpoint('checkpoints', 'epoch_99.cpt')
        self.net.eval()

        @app.route('/')
        def home():
            return render_template('index.html')

        def search(query_vector, k=20):
            top = []
            caption = torch.LongTensor(query_vector)
            if self.conf.gpu_id != -1:
                caption = caption.cuda()
            n_database = len(self.test_db_loader.dataset)
            out_matrix = np.zeros(n_database)
            for images, indexes_q, _ in self.test_db_loader:
                if self.conf.gpu_id != -1:
                    images = images.cuda()
                if self.conf.amp:
                    with autocast():
                        with torch.no_grad():
                            images_feats_out = self.net(images)
                else:
                    with torch.no_grad():
                        images_feats_out = self.net(images)
                captions = caption.repeat(len(images_feats_out), 1, 1)
                captions = captions.squeeze(1)
                if self.conf.amp:
                    with autocast():
                        with torch.no_grad():
                            out = self.net(images_feats_out, captions, query=True)
                else:
                    with torch.no_grad():
                        out = self.net(images_feats_out, captions, query=True)
                out = torch.sigmoid(out)
                out_matrix[indexes_q] = out.squeeze(1).cpu().detach().numpy()
            out_matrix = torch.tensor(out_matrix)
            _, top_k_indexes = out_matrix.topk(k, sorted=True)
            for index in top_k_indexes:
                data = self.test_db_loader.dataset.dataset[index]
                path = data['file_path']
                top.append(path)
            return top

        @app.route('/q')
        def query_page():
            return render_template('query.html')

        @app.route('/query', methods=['get', 'post'])
        def query():
            if request.method == 'POST':
                q = request.form['query']
                result = {'q': q}
                query_vector = self.query_transform(q)
                result['vector'] = query_vector
                top_20 = search(query_vector, k=20)
                result['top'] = top_20
                return render_template('result.html', result=result)
            else:
                return render_template('query.html')

        app.run(debug=self.conf.web_debug, host='0.0.0.0', port=8888)

    def calculate_metrics(self, out_matrix, labels_matrix):
        sorted_indexes = out_matrix.argsort(dim=1)
        r = []
        for k in self.conf.top_k:
            n_corrects = 0
            indexes_k = sorted_indexes[:, :k]
            for index_k, labels in zip(indexes_k, labels_matrix):
                ret = labels[index_k]
                if sum(ret) > 0:
                    n_corrects += 1
            acc = n_corrects / len(labels_matrix)
            self.logger.info(f'top-{k} acc: {acc * 100.0:.2f}%')
            r.append(acc)
        return r

    def save_checkpoint(self, e, checkpoints_dir=None):
        file_name = f'epoch_{e}.cpt'
        if checkpoints_dir is None:
            checkpoints_dir = self.conf.checkpoints_dir
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        file_path = os.path.join(checkpoints_dir, file_name)
        content = {
            'model': self.net.module.state_dict() if config.parallel else self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': e,
            'parallel': self.conf.parallel

        }
        torch.save(content, file_path)
        self.logger.info(f'saved checkpoints: {file_name}')

    def load_checkpoint(self, checkpoints_dir=None, file_name=None):
        if checkpoints_dir is None or file_name is None:
            raise FileNotFoundError('please set checkpoint directory and filename')
        file_path = os.path.join(checkpoints_dir, file_name)
        content = torch.load(file_path)
        if content.get('parallel', True):
            net = torch.nn.parallel.DistributedDataParallel(self.net)
            net.load_state_dict(content['model'])
        else:
            self.net.load_state_dict(content['model'])
        self.logger.info(f'loaded checkpoints from `{file_path}`')


def main():
    d = config.__dict__.copy()
    d.pop('logger')
    j = json.dumps(d, indent=2)
    config.logger.info('\n' + j)
    model = Model(config)

    if config.action == 'process':
        config.logger.info('start to pre-process data')
        model.process()
    elif config.action == 'train':
        config.logger.info('start to train')
        model.train()
    elif config.action == 'test':
        config.logger.info('start to test')
        model.test()
    elif config.action == 'web':
        config.logger.info('start to run a web')
        model.web()
    else:
        raise KeyError(f'No support fot this action: {config.action}')


if __name__ == '__main__':
    main()
