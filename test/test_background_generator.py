import unittest
from simdna import fileProcessing as fp
import simdna
from simdna import synthetic as sn
import numpy as np
from simdna import random
from collections import defaultdict

class TestBackgroundGenerator(unittest.TestCase):

    def test_background_generator(self):
        random.seed(1234)
        np.random.seed(1234)
        seq_length = 100
        #for testing, not biologically realistic
        freqs = {'A': 0.1, 'C': 0.2, 'G': 0.3, 'T': 0.4} 
        embed_in_background = sn.EmbedInABackground(
            sn.ZeroOrderBackgroundGenerator(
             seq_length,
             discreteDistribution=freqs), []) 
        generated_sequences = sn.GenerateSequenceNTimes(
                               embed_in_background, 500).generateSequences() 
        generated_seqs = [seq.seq for seq in generated_sequences]
        char_count = defaultdict(lambda: 0)
        for seq in generated_seqs:
            assert len(seq) == seq_length 
            for char in seq:
                char_count[char] += 1 
        total_chars = sum(char_count.values()) 
        actual_freqs = {val: char_count[val]/float(total_chars)
                     for val in char_count}
        for key in freqs:
            np.testing.assert_almost_equal(actual_freqs[key], freqs[key], 2)
        
