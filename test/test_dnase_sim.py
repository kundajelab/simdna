import unittest
from simdna import fileProcessing as fp
import simdna
from simdna import synthetic as sn
import random

class TestRun(unittest.TestCase):

    def test_run(self):
        dnaseSimulationFileName="temp_dnaseSimulationFile.txt"
        dnaseSimFh = fp.getFileHandle(dnaseSimulationFileName,'w')
        dnaseSimFh.write("sequenceName\tsequence\tmotifs\n")
        dnaseSimFh.write("seq1\tACGTgaTATGATAGCACATGTCGTCAGTACCATGGTCGCCGCTTGCATAGGCAAACATAATTGG\tGATA4_HUMAN.H10MO.B-10,TAL1_known1-30,GATA4_HUMAN.H10MO.B-60\n")
        dnaseSimFh.write("seq2\tACGTGAtaTGATAGCACATGTCGTCAGTACCATGGTCGCCGCTTGCATAGGCAAACATAATTGG\tGATA4_HUMAN.H10MO.B-5,TAL1_known1-35\n")
        dnaseSimFh.write("seq3\tACGTGAtaTGATAGCACATGTCGTCAGTACCATGGTCGCCGCTTGCATAGGCAAACATAATTGG\t"
                         +"GATA_disc1-5,GATA_known1-5,TAL1_known1-5,"
                         +"GATA_disc1-55,GATA_known1-55,TAL1_known1-55\n") #last TAL1 won't get embedded
        dnaseSimFh.write("seq4\tACGTGAtaTGATAGCACATGTCGTCAGTACCATGGTCGCCGCTTGCATAGGCAAACATAATTGG\t"
                         +"GATA_disc1-30,GATA_known1-30,TAL1_known1-30,TAL1_known1-30\n")
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
