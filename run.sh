# python run.py \
# --true_align data/arenas/arenas_edges-mapping-permutation.txt \
# --combined_graph data/arenas/arenas_combined_edges.txt \
# --embmethod xnetMF \
# --alignmethod REGAL \
# # --refinemethod RefiNA 

# python run.py \
# --true_align data/arenas/arenas_edges-mapping-permutation.txt \
# --combined_graph data/arenas/arenas_combined_edges.txt \
# --embmethod xnetMF \
# --alignmethod CONE \
# # --refinemethod RefiNA 

# python run.py \
# --true_align data/arenas/arenas_edges-mapping-permutation.txt \
# --combined_graph data/arenas/arenas_combined_edges.txt \
# --embmethod gwl \
# --alignmethod gwl \
# --refinemethod RefiNA

python run.py \
--true_align data/arenas/arenas_edges-mapping-permutation.txt \
--combined_graph data/arenas/arenas_combined_edges.txt \
--embmethod xnetMF \
--alignmethod IsoRank \
# --refinemethod RefiNA
