from numpy import random
random = random.Random()
random.seed(1)

from pkg_resources import resource_filename
ENCODE_MOTIFS_PATH = resource_filename('simdna.resources', 'encode_motifs.txt.gz')
HOCOMOCO_MOTIFS_PATH = resource_filename('simdna.resources', 'HOCOMOCOv10_HUMAN_mono_homer_format_0.001.motif.gz')
