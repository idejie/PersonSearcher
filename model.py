import torch
import torch.nn as nn
from torchvision.models import vgg16


class Attention(nn.Module):
    def __init__(self, conf):
        super(Attention, self).__init__()
        # generate visual units
        self.fc = nn.Linear(conf.embedding_size, conf.embedding_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rnn_out):
        key = self.fc(rnn_out)
        key = key.transpose(1, 2)
        energy = torch.bmm(rnn_out, key)
        attention = self.softmax(energy)
        # print('dotproct attn', dotproct.shape)  # (batch_size,seq_len,1)
        # (batch_size,seq_len,1) x (batch_size,seq_len,1) = (batch_size,seq_len,1)
        affinity = torch.bmm(attention, rnn_out)  # elementwise multiplication
        return affinity


class Language_Subnet(nn.Module):
    def __init__(self, conf):
        super(Language_Subnet, self).__init__()
        self.word_emb = nn.Embedding(conf.vocab_size, conf.rnn_hidden_size)
        if conf.rnn_layers == 1:
            self.rnn = nn.LSTM(input_size=conf.rnn_hidden_size,
                               hidden_size=conf.embedding_size,
                               num_layers=conf.rnn_layers
                               )
            # self.dropout = nn.Dropout(conf.rnn_dropout)
        else:
            self.rnn = nn.LSTM(input_size=conf.rnn_hidden_size,
                               hidden_size=conf.embedding_size,
                               num_layers=conf.rnn_layers,
                               dropout=conf.rnn_dropout
                               )
        self.conf = conf
        self.attention = Attention(conf)

    def forward(self, captions):
        word_emb = self.word_emb(captions)

        hidden_state_0 = torch.zeros((self.conf.rnn_layers, word_emb.size(1), self.conf.rnn_hidden_size))
        cell_state_0 = torch.zeros((self.conf.rnn_layers, word_emb.size(1), self.conf.rnn_hidden_size))
        if self.conf.gpu_id != -1:
            hidden_state_0 = hidden_state_0.cuda()
            cell_state_0 = cell_state_0.cuda()
        if self.conf.amp:
            from torch.cuda.amp import autocast
            with autocast():
                rnn_out, _ = self.rnn(word_emb.float(), (hidden_state_0.float(), cell_state_0.float()))
        else:
            rnn_out, _ = self.rnn(word_emb, (hidden_state_0, cell_state_0))
        attn_out = self.attention(rnn_out)
        out = torch.sum(attn_out, dim=1)  # sum
        # out = torch.sigmoid(out)
        # print(out.shape, 'out')  # batch_size, 1
        return out


class RankLoss(nn.Module):
    def __init__(self, conf):
        super(RankLoss, self).__init__()
        self.margin = conf.margin
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, query_feats, db_feats, query_ids, db_ids):
        # numpy instead of torch for speeding up
        query_ids = query_ids.cpu().detach().numpy()
        db_ids = db_ids.cpu().detach().numpy()
        cost_q = self.margin
        for i, q in enumerate(query_ids):
            for j, d in enumerate(db_ids):
                if q == d:
                    cost_q += self.cos_sim(query_feats[i].view(1, -1), db_feats[j].view(1, -1))
                else:

                    cost_q -= self.cos_sim(query_feats[i].view(1, -1), db_feats[j].view(1, -1))
        return cost_q


class GNA_RNN(nn.Module):
    def __init__(self, conf):
        super(GNA_RNN, self).__init__()
        # visual sub-network
        self.cnn = vgg16(pretrained=True)
        self.cnn.classifier[-1] = nn.Linear(4096, conf.embedding_size)
        self.language_subnet = Language_Subnet(conf)
        self.img_classifier = nn.Linear(conf.embedding_size, conf.num_classes)
        self.cap_classifier = nn.Linear(conf.embedding_size, conf.num_classes)

    def forward(self, images=None, captions=None):
        try:
            image_feats = self.cnn(images)
            img_out = self.img_classifier(image_feats)
        except:
            image_feats = None
            img_out = None
        try:
            caption_feats = self.language_subnet(captions)
            cap_out = self.cap_classifier(caption_feats)
        except:
            caption_feats = None
            cap_out = None
        return image_feats, img_out, caption_feats, cap_out
