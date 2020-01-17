#!/usr/bin/env python
import os
import sys
import simdna.util.util as util
import simdna.synthetic as sn
import argparse

def do(options):
    outputFileName_core = util.addArguments("EmptyBackground", [
                                                 util.ArgumentToAdd(options.prefix, "prefix"),
                                                 util.ArgumentToAdd(options.seqLength, "seqLength")
                                                 , util.ArgumentToAdd(options.numSeqs, "numSeqs")
                                                 ])
    embedInBackground = sn.EmbedInABackground(
        backgroundGenerator=sn.FirstOrderBackgroundGenerator(seqLength=options.seqLength), embedders=[]
    )
    sequenceSet = sn.GenerateSequenceNTimes(embedInBackground, options.numSeqs)
    sn.printSequences(outputFileName_core+".simdata", sequenceSet, includeFasta=True, includeEmbeddings=True,
                      prefix=options.prefix)
   
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix")
    parser.add_argument("--seqLength", type=int, required=True)
    parser.add_argument("--numSeqs", type=int, required=True)
    options = parser.parse_args()
    do(options) 
