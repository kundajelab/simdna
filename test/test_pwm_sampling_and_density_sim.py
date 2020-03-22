import unittest
from simdna import fileProcessing as fp
import simdna
from simdna import synthetic as sn
import numpy as np
from simdna import random
from collections import defaultdict

class TestBasics(unittest.TestCase):

    def test_density_motif_embedding(self):
        random.seed(1234)
        np.random.seed(1234)
        min_counts = 2
        max_counts = 5
        pseudocount_prob = 0.001
        pwm_name = "CTCF_known1"
        num_sequences = 5000
        loaded_motifs = sn.LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH,
                                   pseudocountProb=pseudocount_prob)
        substring_generator = sn.PwmSamplerFromLoadedMotifs(
            loaded_motifs, pwm_name)
        position_generator = sn.UniformPositionGenerator()
        quantity_generator = sn.UniformIntegerGenerator(min_counts, max_counts)
        embedders = [
            sn.RepeatedEmbedder(
                sn.SubstringEmbedder(
                    sn.ReverseComplementWrapper(
                        substring_generator), position_generator),
                quantity_generator)]
        embed_in_background = sn.EmbedInABackground(
            sn.ZeroOrderBackgroundGenerator(
                500, discreteDistribution={'A':0.3,'C':0.2,
                                                  'G':0.2,'T':0.3}),
            embedders)
        generated_sequences = list(sn.GenerateSequenceNTimes(
            embed_in_background, num_sequences).generateSequences())
        assert len(generated_sequences) == num_sequences

        actual_pwm = np.array([[0.095290, 0.318729, 0.083242, 0.502738],
                         [0.182913, 0.158817, 0.453450, 0.204819],
                         [0.307777, 0.053669, 0.491785, 0.146769],
                         [0.061336, 0.876232, 0.023001, 0.039430],
                         [0.008762, 0.989047, 0.000000, 0.002191],
                         [0.814896, 0.014239, 0.071194, 0.099671],
                         [0.043812, 0.578313, 0.365827, 0.012048],
                         [0.117325, 0.474781, 0.052632, 0.355263],
                         [0.933114, 0.012061, 0.035088, 0.019737],
                         [0.005488, 0.000000, 0.991218, 0.003293],
                         [0.365532, 0.003293, 0.621295, 0.009879],
                         [0.059276, 0.013172, 0.553238, 0.374314],
                         [0.013187, 0.000000, 0.978022, 0.008791],
                         [0.061538, 0.008791, 0.851648, 0.078022],
                         [0.114411, 0.806381, 0.005501, 0.073707],
                         [0.409241, 0.014301, 0.557756, 0.018702],
                         [0.090308, 0.530837, 0.338106, 0.040749],
                         [0.128855, 0.354626, 0.080396, 0.436123],
                         [0.442731, 0.199339, 0.292952, 0.064978]])

        actual_pwm = actual_pwm*(1-pseudocount_prob) + pseudocount_prob/4
        np.testing.assert_almost_equal(np.sum(actual_pwm,axis=-1),1.0,6)
        np.testing.assert_almost_equal(
            actual_pwm,
            np.array(loaded_motifs.getPwm(pwm_name).getRows())) 
        letter_to_index = {'A':0, 'C':1, 'G':2, 'T':3}
        reconstructed_pwm_fwd = np.zeros_like(actual_pwm)
        reconstructed_pwm_rev = np.zeros_like(actual_pwm)
        quantity_distribution = defaultdict(lambda: 0) 
        total_fwd_embeddings = 0.0
        total_rev_embeddings = 0.0
        
        for seq in generated_sequences:
            embeddings = seq.embeddings
            quantity_distribution[len(embeddings)] += 1
            for embedding in embeddings:
                assert (embedding.what.string
                 ==seq.seq[embedding.startPos:
                       embedding.startPos+len(embedding.what.string)])
                if ('revComp' in embedding.what.getDescription()):
                    total_rev_embeddings += 1
                else:
                    total_fwd_embeddings += 1
                for char_idx, char in enumerate(embedding.what.string):
                    if ('revComp' in embedding.what.getDescription()):
                        arr = reconstructed_pwm_rev
                    else:
                        arr = reconstructed_pwm_fwd 
                    arr[char_idx][letter_to_index[char]] += 1

        total_embeddings = total_fwd_embeddings + total_rev_embeddings 
        np.testing.assert_almost_equal(
            total_fwd_embeddings/total_embeddings, 0.5, 2) 

        #normalize each column of reconstructed_pwm
        reconstructed_pwm_fwd = reconstructed_pwm_fwd/total_fwd_embeddings 
        reconstructed_pwm_rev = reconstructed_pwm_rev/total_rev_embeddings 
        np.testing.assert_almost_equal(actual_pwm, reconstructed_pwm_fwd, 2)
        np.testing.assert_almost_equal(actual_pwm,
                                       reconstructed_pwm_rev[::-1,::-1], 2)
       
        #test the quantities of motifs were sampled uniformly  
        for quantity in range(min_counts, max_counts+1):
            np.testing.assert_almost_equal(
             quantity_distribution[quantity]/float(num_sequences),
             1.0/(max_counts-min_counts+1),2)
