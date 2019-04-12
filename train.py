import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import sklearn

from dataloader_retrieval import ArtDatasetRetrieval
from text_encodings import get_text_encoding
from attributes import load_att_class
from model_retrieval_context_kgm import CrossRetrievalContextKGM
from model_retrieval_context_mtl import CrossRetrievalContextMTL
import utils


def save_model(args_dict, state):
    directory = args_dict.dir_model + "%s/"%(args_dict.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'best_model.pth.tar'
    torch.save(state, filename)


def resume(args_dict, model, optimizer):

    best_val = float('Inf')
    args_dict.start_epoch = 0
    if args_dict.resume:
        if os.path.isfile(args_dict.resume):
            print("=> loading checkpoint '{}'".format(args_dict.resume))
            checkpoint = torch.load(args_dict.resume)
            args_dict.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args_dict.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args_dict.resume))
            best_val = float('Inf')

    return best_val, model, optimizer


def trainEpoch(args_dict, train_loader, model, criterion, optimizer, epoch):

    # object to store & plot the losses
    cos_losses = utils.AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Model output
        output = model(input_var[0], input_var[1], input_var[2], input_var[3])

        # Loss
        train_loss = criterion(output[0], output[1], target_var[0].float())
        cos_losses.update(train_loss.data.cpu().numpy(), input[0].size(0))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'vision ({visionLR}) - comment ({textLR}) - class ({classLR})\t'.format(
            epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
            loss=cos_losses, visionLR=optimizer.param_groups[1]['lr'],
            textLR=optimizer.param_groups[0]['lr'], classLR=optimizer.param_groups[2]['lr']))

    # Plot loss
    plotter.plot('cos_loss', 'train', 'CML Loss', epoch, cos_losses.avg)


def valEpoch(args_dict, val_loader, model, criterion, epoch):

    cos_losses = utils.AverageMeter()

    # switch to evaluation mode
    model.eval()
    for batch_idx, (input, target) in enumerate(val_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j], volatile=True).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j], volatile=True))

        # Model output
        with torch.no_grad():
            output = model(input_var[0], input_var[1], input_var[2], input_var[3])

        # Compute loss
        train_loss = criterion(output[0], output[1], target_var[0].float())
        cos_losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # Save embeddings to compute rankings
        if batch_idx==0:
            visual_embs = output[0].data.cpu().numpy()
            text_embs = output[1].data.cpu().numpy()
            img_idxs = target[-2].data.cpu().numpy()
        else:
            visual_embs = np.concatenate((visual_embs,output[0].data.cpu().numpy()),axis=0)
            text_embs = np.concatenate((text_embs,output[1].data.cpu().numpy()),axis=0)
            img_idxs = np.concatenate((img_idxs,target[-2].data.cpu().numpy()),axis=0)

    # Computer retrieval accuracy
    medR, recall = rank(visual_embs, text_embs, img_idxs)

    # Print validation info
    print('Validation set: Average loss: {:.4f}\t'
          'medR {medR:.2f}\t'
          'Recall {recall}'.format(cos_losses.avg, medR=medR, recall=recall))

    # Plot validation results
    plotter.plot('cos_loss', 'test', 'Joint Model Loss', epoch, cos_losses.avg)
    plotter.plot('medR', 'test', 'Joint Model medR', epoch, medR)
    plotter.plot('recall', 'test', 'Joint Model Recall at 10', epoch, recall[10])

    # Return MedR as the validation outcome
    return medR


def rank(img_embeds, text_embeds, ids):

    # Sort indices
    idxs = np.argsort(ids)
    ind = ids[idxs]
    img_emb = img_embeds[idxs,:]
    text_emb = text_embeds[idxs,:]
    N = len(ind)

    # Accuracy variables
    med_rank = []
    recall = {1: 0.0, 5: 0.0, 10: 0.0}

    # Text to image
    for text, idx in zip(text_emb, ind):

        # Cosine similarities between text and all images
        text = np.expand_dims(text, axis=0)
        similarities = np.squeeze(sklearn.metrics.pairwise.cosine_similarity(text, img_emb), axis=0)
        ranking = np.argsort(similarities)[::-1].tolist()

        # position of idx in ranking
        pos = ranking.index(idx)
        if (pos + 1) == 1:
            recall[1] += 1
        if (pos + 1) <= 5:
            recall[5] += 1
        if (pos + 1) <= 10:
            recall[10] += 1

        # store the position
        med_rank.append(pos + 1)

    for i in recall.keys():
        recall[i] = recall[i] / N

    return np.median(med_rank), recall


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


def train_retrieval(args_dict):

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
        assert False, 'Wrong combination of model.'

    # Define model
    if args_dict.model == 'mtl':
        model = CrossRetrievalContextMTL(args_dict, len(vocab_comment), len(vocab_title), num_classes, context_model_path)
    elif args_dict.model == 'kgm':
        model = CrossRetrievalContextKGM(args_dict, len(vocab_comment), len(vocab_title), len(att2i), context_model_path)
    if args_dict.use_gpu:
        model.cuda()

    # Loss and optimizer with 3 parameter groups: vision, context classifier and base
    cosine_loss = nn.CosineEmbeddingLoss(margin=args_dict.margin).cuda()
    class_params = list(map(id, model.classifier.parameters()))
    vision_params = list(map(id, model.resnet.parameters()))
    base_params   = filter(lambda p: id(p) not in vision_params and id(p) not in class_params, model.parameters())
    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.resnet.parameters(), 'lr': args_dict.lr*args_dict.freeVision},
                {'params': model.classifier.parameters(), 'lr': 0}
            ], lr=args_dict.lr*args_dict.freeComment)

    # Resume training if needed
    best_val, model, optimizer = resume(args_dict, model, optimizer)

    # Data transformation for training (with data augmentation) and validation
    train_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.RandomHorizontalFlip(),                  # random horizontal flip (data augmentation)
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataloaders for training and validation
    train_loader = torch.utils.data.DataLoader(
        ArtDatasetRetrieval(args_dict, set = 'train', vocab_comment = vocab_comment, vocab_title = vocab_title,
                            att2i =att2i, transform = train_transforms),
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Training loader with %d samples' % train_loader.__len__())

    val_loader = torch.utils.data.DataLoader(
        ArtDatasetRetrieval(args_dict, set = 'val', vocab_comment = vocab_comment, vocab_title = vocab_title,
                            att2i=att2i, transform = val_transforms),
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Validation loader with %d samples' % val_loader.__len__())

    # Now, let's start the training process!
    print('Start training...')
    pat_track = 0
    for epoch in range(args_dict.start_epoch, args_dict.nepochs):

        # Compute a training epoch
        trainEpoch(args_dict, train_loader, model, cosine_loss, optimizer, epoch)

        # Compute a validation epoch
        accval = valEpoch(args_dict, val_loader, model, cosine_loss, epoch)

        # check patience
        if accval >= best_val:
            pat_track += 1
        else:
            pat_track = 0
        if pat_track >= args_dict.patience:
            args_dict.freeVision = args_dict.freeComment
            args_dict.freeComment = not (args_dict.freeVision)
            optimizer.param_groups[0]['lr'] = args_dict.lr * args_dict.freeComment
            optimizer.param_groups[1]['lr'] = args_dict.lr * args_dict.freeVision
            print 'Initial base params lr: %f' % optimizer.param_groups[0]['lr']
            print 'Initial vision lr: %f' % optimizer.param_groups[1]['lr']
            print 'Initial classifier lr: %f' % optimizer.param_groups[2]['lr']
            args_dict.patience = 3
            pat_track = 0

        # save if it is the best model
        is_best = accval < best_val
        best_val = min(accval, best_val)
        if is_best:
            save_model(args_dict, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'valtrack': pat_track,
                'freeVision': args_dict.freeVision,
                'curr_val': accval,
            })
        print '** Validation: %f (best acc) - %f (current acc) - %d (patience)' % (best_val, accval, pat_track)


def run_train(args_dict):

    # Set seed for reproducibility
    torch.manual_seed(args_dict.seed)
    if args_dict.use_gpu:
        torch.cuda.manual_seed(args_dict.seed)

    # Plots
    global plotter
    plotter = utils.VisdomLinePlotter(env_name=args_dict.name)

    # Main process
    train_retrieval(args_dict)
