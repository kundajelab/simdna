import unittest
from simdna import fileProcessing as fp
import simdna
from simdna import synthetic as sn
import numpy as np
from simdna import random
from collections import defaultdict

class TestPairEmbeddable(unittest.TestCase):

    def test_simple_motif_grammar(self):
        seq_len = 100
        min_sep = 2
        max_sep = 6
        random.seed(1234)
        np.random.seed(1234)
        num_sequences = 4000
        loaded_motifs = sn.LoadedEncodeMotifs(
                         simdna.ENCODE_MOTIFS_PATH,
                         pseudocountProb=0.001)
        motif1_generator = sn.PwmSamplerFromLoadedMotifs(
                            loaded_motifs, "SIX5_known5")
        motif2_generator = sn.PwmSamplerFromLoadedMotifs(
                            loaded_motifs, "ZNF143_known2")
        separation_generator = sn.UniformIntegerGenerator(min_sep,max_sep)
        embedder = sn.EmbeddableEmbedder(
                    sn.PairEmbeddableGenerator(
                     motif1_generator, motif2_generator, separation_generator))
        embed_in_background = sn.EmbedInABackground(
                               sn.ZeroOrderBackgroundGenerator(seq_len),
                               [embedder])
        generated_sequences = sn.GenerateSequenceNTimes(
                        embed_in_background, num_sequences).generateSequences()
        generated_seqs = [seq for seq in generated_sequences]
        separations = defaultdict(lambda: 0) 
        for seq in generated_seqs:
            assert len(seq.seq) == seq_len
            embedding1 = seq.embeddings[0]
            embedding2 = seq.embeddings[1]
            embedding3 = seq.embeddings[2]
            assert len(embedding1.what) == len(embedding1.what.string)
            assert len(embedding2.what) == len(embedding2.what.string)
            assert len(embedding3.what) == (len(embedding1.what)+
                                            len(embedding2.what)+
                                            embedding3.what.separation)
            #testing that the string of the first motif is placed correctly
            assert (seq.seq[
             embedding1.startPos:embedding1.startPos+len(embedding1.what)]
             == embedding1.what.string)
            #testing that the string of the second motif is placed correctly
            assert (seq.seq[
             embedding2.startPos:embedding2.startPos+len(embedding2.what)]
             == embedding2.what.string) 
            #testing that the motifs are placed correctly
            assert ((embedding2.startPos - (embedding1.startPos
                                          + len(embedding1.what.string)))
                     == embedding3.what.separation)
            #test separation is within the right limits 
            assert embedding3.what.separation >= min_sep 
            assert embedding3.what.separation <= max_sep
            #log the separation; will later test distribution
            separations[embedding3.what.separation] += 1

        for possible_sep in range(min_sep, max_sep+1):
            np.testing.assert_almost_equal(
             separations[possible_sep]/float(num_sequences),
             1.0/(max_sep-min_sep+1),2)
             
