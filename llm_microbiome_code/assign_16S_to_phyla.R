# Assign 16S Sequences to Phyla
#
# This script reads 16S rRNA sequences from a FASTA file, installs and loads
# the DADA2 package, and assigns taxonomy using the SILVA database.
#
# Dependencies:
# - devtools
# - dada2 (v1.16)
#
# Input:
# - /path/to/seqs_.07_embed.fasta: FASTA file containing 16S sequences
# - /path/to/silva_nr_v132_train_set.fa.gz: SILVA database for taxonomy assignment
#
# Output:
# - taxa_results.txt: CSV file containing taxonomic assignments
#
# Usage:
# Ensure paths to input files are correct before running the script.
# Run the script in R or RStudio.

seqs <- readLines("/path/to/seqs_.07_embed.fasta")
seqs <- seqs[seq(2, length(seqs) - 1, by=2)]

install.packages("devtools", repos='http://cran.us.r-project.org')
library("devtools")
devtools::install_github("benjjneb/dada2", ref="v1.16")

set.seed(100)
library(dada2); packageVersion("dada2")


taxa <- assignTaxonomy(seqs, "/path/to/silva_nr_v132_train_set.fa.gz", multithread=FALSE)
write.table(taxa, file="taxa_results.txt", sep=",", quote=FALSE, col.names = NA)
