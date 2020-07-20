from numpy import random
from .simdnautil import pwm
from .simdnautil import util
from .simdnautil import dinuc_shuffle

#extend the RandomState to have a random() func,
# for compatibility with np.random
class ExtendedRandomState(random.RandomState):

    def random(self):
        return self.rand(1)[0]

random = ExtendedRandomState()
random.seed(1)

from pkg_resources import resource_filename
ENCODE_MOTIFS_PATH = resource_filename('simdna.resources', 'encode_motifs.txt.gz')
HOCOMOCO_MOTIFS_PATH = resource_filename('simdna.resources', 'HOCOMOCOv10_HUMAN_mono_homer_format_0.001.motif.gz')
