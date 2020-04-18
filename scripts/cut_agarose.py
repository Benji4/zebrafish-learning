import numpy as np
import os

dir = "/disk/scratch/9f" # TODO path to file
outdir = "/disk/scratch/14f" # TODO path to file
in_path = os.path.join(dir, "zebrafish_all.npz") # TODO path to file
out_path = os.path.join(outdir, "zebrafish_all_cut_agarose_85.npz") # TODO path to file

square_size = 85

print("Loading data...")

loaded = np.load(in_path)
X, y = loaded['inputs'], loaded['targets']

print("Cutting agarose...")

# Upper left corner
X[:,:,:square_size,:square_size] *= 0
X[:,:,:square_size,:square_size] += 255

# Lower left corner
X[:,:,-square_size:,:square_size] *= 0
X[:,:,-square_size:,:square_size] += 255

print("Saving compressed file...")
np.savez_compressed(out_path, inputs=X, targets=y)
print("Done. Output file in", out_path)