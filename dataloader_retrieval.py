import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
import pandas as pd
from text_encodings import get_mapped_text
from sklearn.feature_extraction.text import TfidfTransformer
from utils import load_obj, save_obj


class ArtDatasetRetrieval(data.Dataset):

    def __init__(self, args_dict, set, vocab_comment,  vocab_title, att2i, transform = None):

        self.args_dict = args_dict
        self.set = set
        if self.set == 'train':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
            self.mismtch = 0.8
        elif self.set == 'val':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
            self.mismtch = 0
        elif self.set == 'test':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)
            self.mismtch = 0
        df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')

        self.imageurls = list(df['IMAGE_FILE'])
        self.comment_map = get_mapped_text(df, vocab_comment, field='DESCRIPTION')
        self.titles_map = get_mapped_text(df, vocab_title, field='TITLE')
        self.numpairs = len(df) / (1 - self.mismtch)
        self.comw2i = vocab_comment
        self.titw2i = vocab_title
        self.imagefolder = os.path.join(args_dict.dir_dataset, args_dict.dir_images)
        self.transform = transform
        self.att2i = att2i

        # tfidf weights and vectors
        if os.path.exists(args_dict.dir_data + args_dict.tfidf_comments):
            self.tfidf_coms = load_obj(args_dict.dir_data + args_dict.tfidf_comments)
        else:
            self.tfidf_coms = self.get_tfidf(self.comment_map, self.comw2i)
            save_obj(self.tfidf_coms, args_dict.dir_data + args_dict.tfidf_comments)

        if os.path.exists(args_dict.dir_data + args_dict.tfidf_titles):
            self.tfidf_tits = load_obj(args_dict.dir_data + args_dict.tfidf_titles)
        else:
            self.tfidf_tits = self.get_tfidf(self.titles_map, self.titw2i)
            save_obj(self.tfidf_tits, args_dict.dir_data + args_dict.tfidf_titles)

        # get attributes for samples in dataset
        if args_dict.att == 'type':
            self.attributes = list(df['TYPE'])
        elif args_dict.att == 'school':
            self.attributes = list(df['SCHOOL'])
        elif args_dict.att == 'time':
            self.attributes = list(df['TIMEFRAME'])
        elif args_dict.att == 'author':
            self.attributes = list(df['AUTHOR'])

    def get_tfidf(self, text_map, w2i):
        text_onehot = np.zeros((len(text_map),len(w2i)), dtype=np.uint8)
        for i, sentence in enumerate(text_map):
            for j, word in enumerate(sentence):
                text_onehot[i, word] += 1
        tfidf = TfidfTransformer()
        tfidf.fit_transform(text_onehot)
        return tfidf

    def __len__(self):
        return self.numpairs

    def class_from_name(self, vocab, name):
        if vocab.has_key(name):
            idclass= vocab[name]
        else:
            idclass = vocab['UNK']
        return idclass

    def __getitem__(self, index):

        # Pick comment/attributes index --> idx_text
        idx_text = index % len(self.imageurls)

        # Assign if pair is a match or non-match --> target
        if self.set == 'train':
            match = np.random.uniform() > self.mismtch
        else:
            match = True
        target = match and 1 or -1

        # Pick image index: same as idx_text if match, random if non-match --> idx_img
        if target == 1:
            idx_img = idx_text
        else:
            all_idx = range(len(self.imageurls))
            idx_img = np.random.choice(all_idx)
            while idx_img == idx_text:
                idx_img = np.random.choice(all_idx)

        # Load idx_img image & apply transformation --> image
        imagepath = self.imagefolder + self.imageurls[idx_img]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Encode idx_text comment as a tfidf vector --> comment
        comm_map = self.comment_map[idx_text]
        comm_onehot = np.zeros((len(self.comw2i)), dtype=np.uint8)
        for word in comm_map:
            comm_onehot[word] += 1
        comm_tfidf = self.tfidf_coms.transform(comm_onehot.reshape(1, -1))
        comment = torch.FloatTensor(comm_tfidf.toarray())

        # Encode idx_text title as a tfidf vector --> title
        tit_map = self.titles_map[idx_text]
        tit_onehot = np.zeros((len(self.titw2i)), dtype=np.uint8)
        for word in tit_map:
            tit_onehot[word] += 1
        tit_tfidf = self.tfidf_tits.transform(tit_onehot.reshape(1, -1))
        title = torch.FloatTensor(tit_tfidf.toarray())

        # Attribute text
        meta_text_ind = self.class_from_name(self.att2i, self.attributes[idx_text])
        mt_onehot = np.zeros((1, len(self.att2i)), dtype=np.uint8)
        mt_onehot[0, meta_text_ind] = 1
        mt = torch.FloatTensor(mt_onehot)

        # Attribute image
        meta_img_ind = self.class_from_name(self.att2i, self.attributes[idx_img])
        mi_onehot = np.zeros((len(self.att2i)), dtype=np.uint8)
        mi_onehot[meta_img_ind] = 1
        mi = torch.FloatTensor(mi_onehot)

        # Return
        if self.set == 'train':
            return [image, comment, title, mt, mi], [target]
        else:
            return [image, comment, title, mt, mi], [target, idx_img, idx_text]