# python run.py \
# --true_align data/arenas/arenas_edges-mapping-permutation.txt \
# --combined_graph data/arenas/arenas_combined_edges.txt \
# --embmethod xnetMF \
# --alignmethod REGAL \
# --refinemethod RefiNA 

# python run.py \
# --true_align data/arenas/arenas_edges-mapping-permutation.txt \
# --combined_graph data/arenas/arenas_combined_edges.txt \
# --embmethod xnetMF \
# --alignmethod CONE \
# --refinemethod RefiNA \
# --n-update 1

# python run.py \
# --true_align data/arenas/arenas_edges-mapping-permutation.txt \
# --combined_graph data/arenas/arenas_combined_edges.txt \
# --embmethod gwl \
# --alignmethod gwl \
# --refinemethod RefiNA

python run.py \
--true_align data/Magna/Magna-Magna_edges-mapping-permutation.txt \
--combined_graph data/Magna/Magna-Magna_combined_edges.txt \
--embmethod xnetMF \
--alignmethod LREA \
# --refinemethod RefiNA \
# --n-update 1

# python run.py \
# --true_align data/Magna/Magna-Magna_edges-mapping-permutation.txt \
# --combined_graph data/Magna/Magna-Magna_combined_edges.txt \
# --embmethod xnetMF \
# --alignmethod BigAlign \
# --refinemethod RefiNA \
# --n-update 1
