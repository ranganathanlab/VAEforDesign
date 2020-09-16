##########################################################################
# PATHS
#
# These have to be changed to be consistent with user-defined paths. This
# script is tested against the `runAllNBCalcs.sh` scripts, and because the
# script includes a `cd ../` command before running any of the python scripts,
# the base directory is the root of the repository.

# (this directory should contain the file 'pfamseq.txt' from
# ftp://ftp.sanger.ac.uk/pub/databases/Pfam/current_release/database_files/
path2pfamseq = 'data/pfamseq.txt'

# the location of your PDB structures
path2structures = 'data/'

# paths to pymol and needle (EMBOSS) applictaions
path2pymol = '/usr/bin/pymol'
path2needle = '/usr/bin/'

# Also assumes that a folder named 'output/' is in the path
path2output = 'output/'
