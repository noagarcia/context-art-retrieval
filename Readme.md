2022/01/05: Downloading links for the pre-trained models have been updated. Sorry for the wait.

## Context Embeddings for Cross-Modal Retrieval

Pytorch code for the cross-modal retrieval part of our ICMR 2019 paper [Context-Aware Embeddings for Automatic Art Analysis](https://arxiv.org/abs/1904.04985). For the classification part, check [this other repository](https://github.com/noagarcia/context-art-classification). 


### Setup

1. Download dataset from [here](https://noagarcia.github.io/SemArt/).

2. Clone the repository: 
    
    `git clone https://github.com/noagarcia/context-art-retrieval.git`

3. Install dependencies:
    - Python 2.7
    - pytorch (`conda install pytorch=0.4.1 cuda90 -c pytorch`) 
    - torchvision (`conda install torchvision`)
    - visdom (check tutorial [here](https://github.com/noagarcia/visdom-tutorial))
    - pandas (`conda install -c anaconda pandas`)
    - nltk (`conda install -c anaconda nltk`)
    - sklearn (`conda install scikit-learn`)
    
4. Download our pre-trained context-aware models obtained with the [classification](https://github.com/noagarcia/context-art-classification) code and save them into `Models/` folder:
- [MTL](https://drive.google.com/file/d/1KOQgEksQH7sU5syl5Vv-SRilijzae5sP/view?usp=sharing)
- [KGM Type](https://drive.google.com/file/d/1zLdvyy6gSw3ENAhLin6c1-DvnvJ9_CU9/view?usp=sharing)
- [KGM School](https://drive.google.com/file/d/1h5sYNsINQRC4LtdoyMcRi16Ho6evx00n/view?usp=sharing)
- [KGM Timeframe](https://drive.google.com/file/d/1QTsNQbQmQFUgRiWiYjvuPQrgtzL89Ynj/view?usp=sharing)
- [KGM Author](https://drive.google.com/file/d/1dFThWHACyX08mBk8lz-4sC8XhF-VMYam/view?usp=sharing)


### Train

- To train cross-modal retrieval model with MTL context embeddings run:
    
    `python main.py --mode train --model mtl --dir_dataset $semart`
    
- To train cross-modal retrieval model with KGM context embeddings run:
    
    `python main.py --mode train --model kgm --att $attribute --dir_dataset $semart`

Where `$semart` is the path to SemArt dataset and `$attribute` is the classifier type (i.e. `type`, `school`, `time`, or `author`).

### Test

- To test cross-modal retrieval model with MTL context embeddings run:
    
    `python main.py --mode test --model mtl --dir_dataset $semart`
    
- To test cross-modal retrieval model with KGM context embeddings run:
    
    `python main.py --mode test --model kgm --att $attribute --dir_dataset $semart --model_path $model-file`

Where `$semart` is the path to SemArt dataset, `$attribute` is the classifier type (i.e. `type`, `school`, `time`, or `author`), and `$model-file` is the path to the trained model.

You can download our pre-trained cross-modal retrieva models with context embeddings from:
- [MTL Type](https://drive.google.com/file/d/1MZr_kIz65d1nVxb8rpyccg8NIX9S3Yrt/view?usp=sharing)
- [MTL School](https://drive.google.com/file/d/1IepUeQf_Xs7UD1ojhQiloC58DqnOycVE/view?usp=sharing)
- [MTL Timeframe](https://drive.google.com/file/d/1KWwI_OMOBcv6jrLxtrwSwLCaQo0RvTCr/view?usp=sharing)
- [MTL Author](https://drive.google.com/file/d/1qnKGrC3G2Qh4ODCws0rVVXKo8LVF2L7G/view?usp=sharing)
- [KGM Type](https://drive.google.com/file/d/1T2nRDwFugIQUmGce5v0I2q6ApjOWGH7Y/view?usp=sharing)
- [KGM School](https://drive.google.com/file/d/11LxYQuM_KaVswCS5uDv-9JdlaYBhVDa3/view?usp=sharing)
- [KGM Timeframe](https://drive.google.com/file/d/1tFojso0QHwV2mo6hVrhD8f3PQU7oc-x3/view?usp=sharing)
- [KGM Author](https://drive.google.com/file/d/1waBb2gYY6coiXLeQE62oBWBiFof81-hN/view?usp=sharing)


### Results
 
Text-to-Image retrieval results on SemArt dataset:

| Model        | R@1           | R@5  |    R@10    | MedR |
| ------------- |:-------------:| -----:|---------:|--------:|
[CML](https://github.com/noagarcia/SemArt) | 0.164 | 0.384 | 0.505 | 10 |
MTL Type | 0.145 | 0.358 | 0.474 | 12 |
MTL School | 0.196 | 0.428 | 0.536 | 8 |
MTL TF | 0.171 | 0.394 | 0.525 | 9 |
MTL Author | 0.232 | 0.452 | 0.567 | 7 |
KGM Type | 0.152 | 0.367 | 0.506 | 10 |
KGM School | 0.162 | 0.371 | 0.483 | 12 |
KGM TF | 0.175 | 0.399 | 0.506 | 10 | 
KGM Author | **0.247** | **0.477** | **0.581** | **6**  |


### Citation

```
@InProceedings{Garcia2017Context,
   author    = {Noa Garcia and Benjamin Renoust and Yuta Nakashima},
   title     = {Context-Aware Embeddings for Automatic Art Analysis},
   booktitle = {Proceedings of the ACM International Conference on Multimedia Retrieval},
   year      = {2019},
}
``` 

[1]: http://researchdata.aston.ac.uk/380/
[2]: https://github.com/facebookresearch/visdom
