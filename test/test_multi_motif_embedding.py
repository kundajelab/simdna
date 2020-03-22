import unittest
from simdna import fileProcessing as fp
import simdna
from simdna import synthetic as sn
import numpy as np
from simdna import random
from collections import defaultdict
np.random.seed(1234)
random.seed(1234)

class TestBasics(unittest.TestCase):

    def test_multi_motif_embedding(self):
   
        motif_names = ["CTCF_known1", "IRF_known1",
                       "SPI1_known1", "CTCF_known2", "CTCF_disc1"] 
        loaded_motifs = sn.LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH,
                                   pseudocountProb=0.001)
        position_generator = sn.UniformPositionGenerator()
        embedders = [sn.SubstringEmbedder(sn.PwmSamplerFromLoadedMotifs(
                     loaded_motifs, motif_name),
                     position_generator, name=motif_name)
                     for motif_name in motif_names]
        min_selected_motifs = 1
        max_selected_motifs = 4
        quantity_generator = sn.UniformIntegerGenerator(min_selected_motifs,
                                                     max_selected_motifs)
        combined_embedder = [sn.RandomSubsetOfEmbedders(
                             quantity_generator, embedders)]
        embed_in_background = sn.EmbedInABackground(
            sn.ZeroOrderBackgroundGenerator(
             300, discreteDistribution={'A':0.3, 'C':0.2, 'G':0.2, 'T':0.3}),
            combined_embedder)
        generated_sequences = tuple(sn.GenerateSequenceNTimes(
            embed_in_background, 8000).generateSequences())
        sequence_arr = np.array([generated_seq.seq for
                                 generated_seq in generated_sequences])
        label_generator = sn.IsInTraceLabelGenerator(np.array(motif_names))
        y = np.array([label_generator.generateLabels(generated_seq)
                      for generated_seq in generated_sequences]).astype(bool)
        embedding_arr = [generated_seq.embeddings for generated_seq in generated_sequences]

        num_embeddings_count = defaultdict(lambda: 0)
        for seq, labels, embeddings, generated_seq in zip(sequence_arr, y, embedding_arr, generated_sequences):
            motifs_embedded = set()
            num_embeddings_count[len(embeddings)] += 1
            for embedding in embeddings:
                #assert that the string selected is correct
                assert embedding.what.string ==\
                        seq[embedding.startPos:
                            (embedding.startPos+len(embedding.what.string))]
                motifs_embedded.add(embedding.what.getDescription()) 
            assert len(motifs_embedded) == len(embeddings) #non-redundant
            for (motif_idx, motif_name) in enumerate(motif_names):
                if motif_name in motifs_embedded:
                    assert labels[motif_idx]==True
                else:
                    assert labels[motif_idx]==False
        
        #assert that the num selected is drawn correctly from a uniform dist
        for num_selected_motifs in range(min_selected_motifs,
                                         max_selected_motifs+1): 
            np.testing.assert_almost_equal(
        num_embeddings_count[num_selected_motifs]/float(len(sequence_arr)),
        1.0/(max_selected_motifs-min_selected_motifs+1),2)
        #there also shouldn't be a preference for any one motif over others
        np.testing.assert_almost_equal(
            np.sum(y,axis=0)/float(np.sum(y)), 1.0/len(motif_names), 2)
