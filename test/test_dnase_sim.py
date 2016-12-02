import unittest
from simdna import fileProcessing as fp
import simdna
from simdna import synthetic as sn

class TestRun(unittest.TestCase):

    def test_run(self):
        dnaseSimulationFileName="temp_dnaseSimulationFile.txt"
        dnaseSimFh = fp.getFileHandle(dnaseSimulationFileName,'w')
        dnaseSimFh.write("sequenceName\tsequence\tmotifs\n")
        dnaseSimFh.write("seq1\tACGTgaTATGATAGCACATGTCGTCAGTACCATGGTCGCCGCTTGCATAGGCAAACATAATTGG\tGATA4_HUMAN.H10MO.B-10,TAL1_known1-30,GATA4_HUMAN.H10MO.B-60\n")
        dnaseSimFh.write("seq2\tACGTGAtaTGATAGCACATGTCGTCAGTACCATGGTCGCCGCTTGCATAGGCAAACATAATTGG\tGATA4_HUMAN.H10MO.B-5,TAL1_known1-35\n")
        dnaseSimFh.close()

        dnaseSimulation = sn.DnaseSimulation(
            dnaseSimulationFile=dnaseSimulationFileName,
            loadedMotifs=sn.LoadedEncodeMotifs(
                            simdna.ENCODE_MOTIFS_PATH,
                            pseudocountProb=0.001).addMotifs(
                                sn.LoadedHomerMotifs(                                  
                                simdna.HOCOMOCO_MOTIFS_PATH,                                    
                                pseudocountProb=0.000)),
            shuffler=sn.DinucleotideShuffler())
        sn.printSequences("temp_dnaseSimulation.simdata", dnaseSimulation,       
                             includeFasta=False, includeEmbeddings=True,         
                             prefix=None)
