import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='test', type=str, help='Mode (train | test)')
    parser.add_argument('--model', default='kgm', type=str, help='Model (mtl | kgm). mlt for multitask learning model. kgm for knowledge graph model.' )
    parser.add_argument('--att', default='author', type=str, help='Attribute classifier (type | school | time | author) (only kgm model).')

    # Directories
    parser.add_argument('--dir_images', default='Images/')
    parser.add_argument('--dir_data', default='Data/')
    parser.add_argument('--dir_dataset', default='')
    parser.add_argument('--dir_model', default='Models/', help='Path to project data')

    # Files
    parser.add_argument('--csvtrain', default='semart_train.csv', help='Training set data file')
    parser.add_argument('--csvval', default='semart_val.csv', help='Dataset val data file')
    parser.add_argument('--csvtest', default='semart_test.csv', help='Dataset test data file')
    parser.add_argument('--vocab_comments', default='w2i_comments.pckl')
    parser.add_argument('--vocab_titles', default='w2i_titles.pckl')
    parser.add_argument('--vocab_type', default='type2ind.pckl', help='Type classes file')
    parser.add_argument('--vocab_school', default='school2ind.pckl', help='Author classes file')
    parser.add_argument('--vocab_time', default='time2ind.pckl', help='Timeframe classes file')
    parser.add_argument('--vocab_author', default='author2ind.pckl', help='Author classes file')
    parser.add_argument('--tfidf_comments', default='tfidf_comments.pckl', help='File with the weights of the tfidf model')
    parser.add_argument('--tfidf_titles', default='tfidf_titles.pckl', help='File with the weights of the tfidf model')

    # Context models
    parser.add_argument('--mtl_model', default='Models/best-mtl-model.pth.tar', type=str)
    parser.add_argument('--kgm_type_model', default='Models/best-kgm-type-model.pth.tar', type=str)
    parser.add_argument('--kgm_school_model', default='Models/best-kgm-school-model.pth.tar', type=str)
    parser.add_argument('--kgm_time_model', default='Models/best-kgm-time-model.pth.tar', type=str)
    parser.add_argument('--kgm_author_model', default='Models/best-kgm-author-model.pth.tar', type=str)

    # Training
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--freeVision', default=False, type=bool)
    parser.add_argument('--freeComment', default=True, type=bool)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--nepochs', default=50, type=int)

    # Model params
    parser.add_argument('--margin', default=0.1, type=float)
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--patience', default=1, type=int)

    # Test
    parser.add_argument('--model_path', default='Models/best-retrieval-kgm-author.pth.tar', type=str)
    parser.add_argument('--path_results', default='Results/', type=str)
    parser.add_argument('--no_cuda', action='store_true')

    return parser