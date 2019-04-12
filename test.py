import os
import numpy as np
import sklearn
import torch
from torchvision import transforms

import utils
from dataloader_retrieval import ArtDatasetRetrieval
from model_retrieval_context_kgm import CrossRetrievalContextKGM
from model_retrieval_context_mtl import CrossRetrievalContextMTL
from text_encodings import get_text_encoding
from attributes import load_att_class


def measure_test_acc(img_emb, text_emb, text_ind):

    # Sort indices
    idxs = np.argsort(text_ind)
    text_ind = text_ind[idxs]
    img_emb = img_emb[idxs,:]
    text_emb = text_emb[idxs,:]
    N = len(text_ind)

    # Accuracy variables
    med_rank_t2i, med_rank_i2t = [], []
    recall_t2i = {1: 0.0, 5: 0.0, 10: 0.0}
    recall_i2t = {1: 0.0, 5: 0.0, 10: 0.0}

    # Text to image
    for text, idx in zip(text_emb, text_ind):

        # Cosine similarities between text and all images
        text = np.expand_dims(text, axis=0)
        similarities = np.squeeze(sklearn.metrics.pairwise.cosine_similarity(text, img_emb), axis=0)
        ranking_t2i = np.argsort(similarities)[::-1].tolist()

        # position of idx in ranking
        pos = ranking_t2i.index(idx)
        if (pos + 1) == 1:
            recall_t2i[1] += 1
        if (pos + 1) <= 5:
            recall_t2i[5] += 1
        if (pos + 1) <= 10:
            recall_t2i[10] += 1

        # store the position
        med_rank_t2i.append(pos + 1)

    # Image to text
    for img, idx in zip(img_emb, text_ind):

        # Cosine similarities between text and all images
        img = np.expand_dims(img, axis=0)
        similarities = np.squeeze(sklearn.metrics.pairwise.cosine_similarity(img, text_emb), axis=0)
        ranking_i2t = np.argsort(similarities)[::-1].tolist()

        # position of idx in ranking
        pos2 = ranking_i2t.index(idx)
        if (pos2 + 1) == 1:
            recall_i2t[1] += 1
        if (pos2 + 1) <= 5:
            recall_i2t[5] += 1
        if (pos2 + 1) <= 10:
            recall_i2t[10] += 1

        # store the position
        med_rank_i2t.append(pos2 + 1)

    for i in recall_t2i.keys():
        recall_t2i[i] = recall_t2i[i] / N

    for i in recall_i2t.keys():
        recall_i2t[i] = recall_i2t[i] / N

    print('------------ Test Accuracy -------------')
    print "Median t2i", np.median(med_rank_t2i)
    print "Recall t2i", recall_t2i
    print "Median i2t", np.median(med_rank_i2t)
    print "Recall i2t", recall_i2t
    print('----------------------------------------')


def load_vocabularies(args_dict):

    if os.path.isfile(args_dict.dir_data + args_dict.vocab_comments):
        vocab_comment = utils.load_obj(args_dict.dir_data + args_dict.vocab_comments)
        print('Comment vocabulary loaded from %s' % (args_dict.dir_data + args_dict.vocab_comments))
    else:
        print("Creating comments vocabulary... ")
        vocab_comment, _ = get_text_encoding(os.path.join(args_dict.dir_dataset, args_dict.csvtrain), 10000, 'DESCRIPTION')
        utils.save_obj(vocab_comment, args_dict.dir_data + args_dict.vocab_comments)

    if os.path.isfile(args_dict.dir_data + args_dict.vocab_titles):
        vocab_title = utils.load_obj(args_dict.dir_data + args_dict.vocab_titles)
        print('Titles vocabulary loaded from %s' % (args_dict.dir_data + args_dict.vocab_titles))
    else:
        print("Creating titles vocabulary... ")
        vocab_title, _ = get_text_encoding(os.path.join(args_dict.dir_dataset, args_dict.csvtrain), -1, 'TITLE')
        utils.save_obj(vocab_title, args_dict.dir_data + args_dict.vocab_titles)

    return vocab_comment, vocab_title


def extract_test_features(args_dict):

    # Load data
    vocab_comment, vocab_title = load_vocabularies(args_dict)
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    if args_dict.model == 'kgm' and args_dict.att == 'type':
        att2i = type2idx
        context_model_path = args_dict.kgm_type_model
    elif args_dict.model == 'kgm' and args_dict.att == 'school':
        att2i = school2idx
        context_model_path = args_dict.kgm_school_model
    elif args_dict.model == 'kgm' and args_dict.att == 'time':
        att2i = time2idx
        context_model_path = args_dict.kgm_time_model
    elif args_dict.model == 'kgm' and args_dict.att == 'author':
        att2i = author2idx
        context_model_path = args_dict.kgm_author_model
    elif args_dict.model == 'mtl':
        num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
        context_model_path = args_dict.mtl_model
        if args_dict.att == 'type':
            att2i = type2idx
        elif args_dict.att == 'school':
            att2i = school2idx
        elif args_dict.att == 'time':
            att2i = time2idx
        elif args_dict.att == 'author':
            att2i = author2idx
    else:
        assert False, 'Wrong combination of model and att parameters.'

    # Define model
    if args_dict.model == 'mtl':
        model = CrossRetrievalContextMTL(args_dict, len(vocab_comment), len(vocab_title), num_classes, context_model_path)
    elif args_dict.model == 'kgm':
        model = CrossRetrievalContextKGM(args_dict, len(vocab_comment), len(vocab_title), len(att2i), context_model_path)
    if args_dict.use_gpu:
        model.cuda()

    # Load best model
    print("=> loading checkpoint '{}'".format(args_dict.model_path))
    checkpoint = torch.load(args_dict.model_path)
    args_dict.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args_dict.model_path, checkpoint['epoch']))

    # Data transformation for test
    test_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for test
    test_loader = torch.utils.data.DataLoader(
        ArtDatasetRetrieval(args_dict, set = 'test', vocab_comment = vocab_comment, vocab_title = vocab_title,
                            att2i =att2i, transform = test_transforms),
        batch_size=args_dict.batch_size, shuffle=False, pin_memory=(not args_dict.no_cuda), num_workers=args_dict.workers)
    print('Test loader with %d samples' % test_loader.__len__())

    # Switch to evaluation mode & compute test
    model.eval()
    for i, (input, target) in enumerate(test_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        output = model(input_var[0], input_var[1], input_var[2], input_var[3])

        # Store embeddings
        if i==0:
            visual_embs = output[0].data.cpu().numpy()
            text_embs = output[1].data.cpu().numpy()
            img_idxs = target[-2].data.cpu().numpy()
        else:
            visual_embs = np.concatenate((visual_embs,output[0].data.cpu().numpy()),axis=0)
            text_embs = np.concatenate((text_embs,output[1].data.cpu().numpy()),axis=0)
            img_idxs = np.concatenate((img_idxs,target[-2].data.cpu().numpy()),axis=0)

    # Save embeddings
    utils.save_obj(visual_embs, args_dict.path_results + 'img_embeds.pkl')
    utils.save_obj(text_embs, args_dict.path_results + 'text_embeds.pkl')
    utils.save_obj(img_idxs, args_dict.path_results + 'ids.pkl')
    return visual_embs, text_embs, img_idxs


def run_test(args_dict):

    # Create directories
    args_dict.path_results += args_dict.name + '/'
    if not os.path.exists(args_dict.path_results):
        os.makedirs(args_dict.path_results)

    # Run test and print accuracy
    img_emd, text_emd, ind = extract_test_features(args_dict)
    measure_test_acc(img_emd, text_emd, ind)



