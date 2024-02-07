# GALib：A More User-Friendly and Highly Versatile Python Library for Graph Alignment
Introducing Graph Alignment Python Library (GALib): 
GALib is a lightweight, user-friendly, and highly versatile Python library for diverse unrestricted graph alignment algorithms, which calls for finding a matching between the nodes of one graph and those of another graph, in a way that they correspond to each other by some fitness measure. With GALib, you can effortlessly align graphs and leverage its extensive range of functionalities.

# Overview

We have developed GALib, an open-source library for network alignment, using Python and PyTorch. To facilitate the integration of representative models and enable third-party developers to extend it according to their specific requirements, we abstract this framework into two main components: Encoder and Decoder. The Encoder takes the structural information of two graphs as input, with the option to include label information, and outputs the similarity matrices of the two graphs. The Decoder processes the obtained similarity matrices to derive the final alignment matrix. The Decoder is followed by a search module for alignment tasks. To facilitate the extension to other downstream tasks such as multimodal alignment and graph representation learning, we extract the search module from the Decoder component, allowing third-party developers to modify it as needed. The output of the search module is the final evaluation metric results (Hits@k, MRR, and MNC).

## Algorithms

We integrate eleven representative network-alignment as Encoder and deocder. Their papers and the original codes are given in the following table.

|   Encoder   |     Paper     |     Publish     |
|:--------:|:------------------------------------:|:------------:|
|  Isorank     |         Global alignment of multiple protein interaction networks with application to functional orthology detection           |    [PNAS'2008](https://www.pnas.org/content/105/35/12763)    |
|  NSD       |          Network Similarity Decomposition (NSD): A Fast and Scalable Approach to Network Alignment            |    [IEEE'2012](https://ieeexplore.ieee.org/document/5975146)    |
|  Big-Align  |         BIG-ALIGN: Fast Bipartite Graph Alignment           |  [IEEE'2013](https://ieeexplore.ieee.org/abstract/document/6729523)  |
|  FINAL   |         FINAL: Fast Attributed Network Alignment           |  [KDD '2016](https://dl.acm.org/doi/abs/10.1145/2939672.2939766)  |
|  Regal     |         REGAL: Representation Learning-based Graph Alignment           |    [CIKM '2018](https://dl.acm.org/doi/10.1145/3269206.3271788)    |
|  LREA        |         Low Rank Spectral Network Alignment           |    [WWW '2018](https://dl.acm.org/doi/10.1145/3178876.3186128)    |
|  GWL  |         Gromov-Wasserstein Learning for Graph Matching and Node Embedding          |  [arXiv'2019](https://arxiv.org/abs/1901.06003)  |
|  CΟΝΕ   |         CONE-Align: Consistent Network Alignment with Proximity-Preserving Node Embedding           |  [CIKM '2020](https://dl.acm.org/doi/10.1145/3340531.3412136)  |
| Grampa        |         Spectral graph matching and regularized quadratic relaxations: algorithm and theory           | [ICML'2020](https://dl.acm.org/doi/abs/10.5555/3524938.3525218) |
|  Grasp        |         GRASP: Graph Alignment Through Spectral Signatures           |    [APWeb-WAIM'2021](https://link.springer.com/chapter/10.1007/978-3-030-85896-4_4)    |
| B-Grasp        |         GRASP: Scalable Graph Alignment by Spectral Corresponding Functions           | [TKDD'2023](https://dl.acm.org/doi/full/10.1145/3561058) |
|   **Decoder**   |     **Paper**     |     **Publish**     |
|  RefiNA  |         Refining Network Alignment to Improve Matched Neighborhood Consistency           |  [SDM'2021](https://epubs.siam.org/doi/abs/10.1137/1.9781611976700.20)  |
|  CAPER  |         CAPER: Coarsen, Align, Project, Refine - A General Multilevel Framework for Network Alignment           |  [arXiv'2022](https://arxiv.org/abs/2208.10682)  |
|  Greed-Match  |         -           |  [-](-)   |
|  Sinkhorn  |        From Alignment to Assignment: Frustratingly Simple Unsupervised Entity Alignment         | [EMNLP'2021](https://arxiv.org/abs/2109.02363)  |

## Datasets

We have standardized the format of the experimental data, and more information about the datasets can be accessed at [CAPER](https://github.com/GemsLab/CAPER).

| Name | Nodes | Edges  | Description
|:--------:|:-------:|:-------:|:--------:|
| [Arenas](https://dl.acm.org/doi/abs/10.1145/2487788.2488173) | 1,133 | 5,451 | communication network
| [Hamsterster](https://dl.acm.org/doi/abs/10.1145/2487788.2488173) | 2,426 | 16,613 | social network
| [Facebook](http://snap.stanford.edu/data/) | 4,039 | 88,234 | social network
| [Magna](https://academic.oup.com/bioinformatics/article/30/20/2931/2422208?login=false) | 1,004 | 8,323 | protein-protein interaction

# Usage

This section provides instructions on obtaining a copy of the library, as well as guidance on installing and running it on your local machine for development and testing purposes. Additionally, it offers an overview of the package structure of the source code.

## Package Description

```
src/
├── GALib/
│   ├── encoder/: package of the implementations for existing representative network alignment approaches
│   ├── decoder/: package of the implementations for refining simiarlity matrix approaches
│   ├── matcher/: package of the implementations for the search utils of framework, search module, and their interaction
│   ├── input/: package of the components for datasets process
│   ├── utils/: package of the components for this framework
```

## Requirements

- Python(3.7)
- numpy(1.20.3) 
- scipy(1.7.3) 
- networkx(1.11) 
- pickle 
- scikit-learn(0.24)
- sacred(0.8.2) 
- theano (1.0.5) 
- pymanopt(0.2.5) 
- pandas(1.1.3) 

## Get Started

1. Run an implemented model

```bash
chmod u+x run.sh
./run.sh
```

2. Modify and run a script as follows (examples are in `run/`):

```
python main.py \
--true_align data/arenas/arenas_edges-mapping-permutation.txt \
--combined_graph data/arenas/arenas_combined_edges.txt \
--embmethod xnetMF \
--alignmethod CONE \
--refinemethod RefiNA 
```

3. If you want to adjust the architecture of the model yourself, or set the parameters of each part of Encoder or Decoder, you can do so on the 'main.py' as follows (take CONE as Encoder and RefiNA as Decoder, search by greed_match):
```python
from encoder.CONE.CONE import CONE
from decoder.RefiNA.RefiNA import RefiNA

encoder = CONE(adjA, adjB, dim=64,window=10,negative=1.0,niter_init=10,reg_init=1.0, \
                 lr=1.0,bsz=10,nepoch=5,embsim="euclidean",numtop=10,reg_align=0.05,niter_align=10)
alignment_matrix = encoder.align()

decoder = RefiNA(alignment_matrix, adjA, adjB, token_match=1, n_update=1, iter=100)
alignment_matrix = decoder.refine_align()

score, _ = refina_utils.score_alignment_matrix(alignment_matrix, topk = 1, true_alignments = true_align)
```

## Acknowledgement
The Codebase is built upon the following work -
- [thanhtrunghuynh93](https://github.com/thanhtrunghuynh93/networkAlignment)
- [constantinosskitsas](https://github.com/constantinosskitsas/Framework_GraphAlignment)
- [CAPER](https://github.com/GemsLab/CAPER)

We appreciate them and many other related works for their open-source contributions.

