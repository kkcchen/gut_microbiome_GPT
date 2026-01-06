import numpy as np
import anndata as ad

# --- load your data ---
a = ad.read_h5ad("/project/aip-rahulgk/gutmodel/remove_redundant_oct24/finetune_train.h5ad")
b = ad.read_h5ad("/project/aip-rahulgk/gutmodel/remove_redundant_oct24/finetune_test.h5ad")
c = ad.read_h5ad("/project/aip-rahulgk/gutmodel/remove_redundant_oct24/pretrain.h5ad")

# --- check that taxa order is identical ---
same_a = np.array_equal(a.var_names, c.var_names)
same_b = np.array_equal(b.var_names, c.var_names)

print("a and c taxa identical:", same_a)
print("b and c taxa identical:", same_b)

# --- if identical, copy varm['taxonomy'] ---
if same_a and same_b:
    a.varm['taxonomy'] = c.varm['taxonomy'].copy()
    b.varm['taxonomy'] = c.varm['taxonomy'].copy()
    print("Copied taxonomy to both a and b.")
else:
    raise ValueError("Taxa orders differ — cannot safely copy taxonomy.")

# --- save back ---
a.write_h5ad("/project/aip-rahulgk/gutmodel/remove_redundant_oct24/finetune_train_with_taxonomy.h5ad", compression='gzip')
b.write_h5ad("/project/aip-rahulgk/gutmodel/remove_redundant_oct24/finetune_test_with_taxonomy.h5ad", compression='gzip')

print("Saved updated .h5ad files successfully.")
