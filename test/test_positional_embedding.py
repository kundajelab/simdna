import unittest
from simdna import fileProcessing as fp
import simdna
from simdna import synthetic as sn
import numpy as np
from simdna import random
from collections import defaultdict
random.seed(1234)
np.random.seed(1234)

class TestPositionalEmbedding(unittest.TestCase):

    def test_uniform_positions(self):
        pseudocount_prob = 0.001
        pwm_name = "CTCF_known1"
        num_sequences = 10000
        sequence_length = 50
        loaded_motifs = sn.LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH,
                                   pseudocountProb=pseudocount_prob)
        substring_generator = sn.PwmSamplerFromLoadedMotifs(
            loaded_motifs, pwm_name)
        position_generator = sn.UniformPositionGenerator()
        embedders = [
                sn.SubstringEmbedder(substring_generator, position_generator)]
        embed_in_background = sn.EmbedInABackground(
            sn.ZeroOrderBackgroundGenerator(
                sequence_length, discreteDistribution={'A':0.3,'C':0.2,
                                                  'G':0.2,'T':0.3}),
            embedders)
        generated_sequences = list(sn.GenerateSequenceNTimes(
            embed_in_background, num_sequences).generateSequences())

        motif_length = len(loaded_motifs.getPwm(pwm_name).getRows())
        start_pos_count = np.zeros(sequence_length-motif_length+1)

        for seq in generated_sequences:
            assert len(seq.seq)==sequence_length
            embeddings = seq.embeddings
            for embedding in embeddings:
                assert (embedding.what.string
                 ==seq.seq[embedding.startPos:
                       embedding.startPos+len(embedding.what.string)])
                start_pos_count[embedding.startPos] += 1

        start_pos_count = start_pos_count/float(len(generated_sequences))
        np.testing.assert_almost_equal(start_pos_count,
                                       1.0/len(start_pos_count), 2)

    def test_central_positions(self):
        pseudocount_prob = 0.001
        pwm_name = "CTCF_known1"
        num_sequences = 10000
        sequence_length = 50
        loaded_motifs = sn.LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH,
                                   pseudocountProb=pseudocount_prob)
        substring_generator = sn.PwmSamplerFromLoadedMotifs(
            loaded_motifs, pwm_name)
        position_generator = sn.InsideCentralBp(30)
        embedders = [
                sn.SubstringEmbedder(substring_generator, position_generator)]
        embed_in_background = sn.EmbedInABackground(
            sn.ZeroOrderBackgroundGenerator(
                sequence_length, discreteDistribution={'A':0.3,'C':0.2,
                                                  'G':0.2,'T':0.3}),
            embedders)
        generated_sequences = list(sn.GenerateSequenceNTimes(
            embed_in_background, num_sequences).generateSequences())

        motif_length = len(loaded_motifs.getPwm(pwm_name).getRows())
        start_pos_count = np.zeros(sequence_length-motif_length+1)

        for seq in generated_sequences:
            assert len(seq.seq)==sequence_length
            embeddings = seq.embeddings
            for embedding in embeddings:
                assert (embedding.what.string
                 ==seq.seq[embedding.startPos:
                       embedding.startPos+len(embedding.what.string)])
                start_pos_count[embedding.startPos] += 1

        start_pos_count = start_pos_count/float(len(generated_sequences))
        #the *1.0 is for conversion to float
        expected_start_pos_count = np.zeros_like(start_pos_count).astype("float32")
        #expect motif to be embedded only in the central 40bp
        expected_start_pos_count[10:(40-motif_length+1)] = 1.0/(30.0-motif_length+1)
        np.testing.assert_almost_equal(start_pos_count,
                                       expected_start_pos_count, 2)
