import torch
import torch.nn as nn
from torchvision import models
from model_mtl import MTL


class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()

    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)


class CrossRetrievalContextMTL(nn.Module):
    def __init__(self, args_dict, comments_vocab_size, titles_vocab_size, att_size, context_model):
        super(CrossRetrievalContextMTL, self).__init__()

        self.args_dict = args_dict
        self.table = TableModule()
        if self.args_dict.att == 'type':
            self.n = att_size[0]
        if self.args_dict.att == 'school':
            self.n = att_size[1]
        if self.args_dict.att == 'time':
            self.n = att_size[2]
        if self.args_dict.att == 'author':
            self.n = att_size[3]

        # Context classifier
        classifier = MTL(att_size)
        checkpoint = torch.load(context_model)
        classifier.load_state_dict(checkpoint['state_dict'])
        self.classifier = classifier
        for param in classifier.parameters():
            param.requires_grad = False

        # Visual embeddings
        resnet = models.resnet50(pretrained=True)
        self.resnet = resnet
        self.visual_embedding = nn.Sequential(
            nn.Linear(1000 + self.n, args_dict.emb_size),
            nn.Tanh(),
        )

        # Text embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(comments_vocab_size + titles_vocab_size + self.n, args_dict.emb_size),
            nn.Tanh(),
        )

        self.end2end = False

    def forward(self, img, com, tit, att_t):
        # inputs:
        #     - img: image
        #     - com: comment
        #     - tit: title
        #     - att_t: attribute from text

        # Attribute from context classifier
        [type, school, tf, author] = self.classifier(img)
        if self.args_dict.att == 'type':
            outclass = type
        if self.args_dict.att == 'school':
            outclass = school
        if self.args_dict.att == 'time':
            outclass = tf
        if self.args_dict.att == 'author':
            outclass = author
        _, pred = torch.max(outclass, 1)
        pred_ = torch.unsqueeze(pred, 1)
        att_i = torch.cuda.FloatTensor(pred_.shape[0], self.n).zero_()
        att_i.scatter_(1, pred_, 1)

        # Visual embedding
        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = torch.squeeze(self.table([visual_emb, att_i], 1), 1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)

        # Text embedding
        text_emb = torch.squeeze(self.table([com, tit, att_t],2),1)
        text_emb = self.text_embedding(text_emb)
        text_emb = norm(text_emb)

        # Output
        return [visual_emb, text_emb, outclass]
