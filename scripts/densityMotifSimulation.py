#!/usr/bin/env python
import os
import sys
import simdna
import simdna.util as util
import simdna.synthetic as synthetic
import argparse

def do(options):
    if (options.seed is not None):
        import numpy as np
        np.random.seed(options.seed) 
        import random
        random.seed(options.seed)
        
    outputFileName_core = util.addArguments("DensityEmbedding",
                        [util.ArgumentToAdd(options.prefix, "prefix"),
                         util.BooleanArgument(options.bestHit, "bestHit"),
                         util.ArrArgument(options.motifNames, "motifs"),
                         util.ArgumentToAdd(options.min_motifs, "min"),
                         util.ArgumentToAdd(options.max_motifs, "max"),
                         util.ArgumentToAdd(options.mean_motifs, "mean"),
                         util.ArgumentToAdd(options.zero_prob, "zeroProb"),
                         util.ArgumentToAdd(options.seqLength, "seqLength"),
                         util.ArgumentToAdd(options.numSeqs, "numSeqs")])
    
    loadedMotifs = synthetic.LoadedEncodeMotifs(options.pathToMotifs, pseudocountProb=0.001)
    Constructor = synthetic.BestHitPwmFromLoadedMotifs if options.bestHit else synthetic.PwmSamplerFromLoadedMotifs   
    embedInBackground = synthetic.EmbedInABackground(
        backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=options.seqLength) 
        , embedders=[
            synthetic.RepeatedEmbedder(
            synthetic.SubstringEmbedder(
                #synthetic.ReverseComplementWrapper(
                substringGenerator=Constructor(
                    loadedMotifs=loadedMotifs,motifName=motifName)
                #),
                ,positionGenerator=synthetic.UniformPositionGenerator()),
            quantityGenerator=synthetic.ZeroInflater(synthetic.MinMaxWrapper(
                synthetic.PoissonQuantityGenerator(options.mean_motifs),
                theMax=options.max_motifs, theMin=options.min_motifs), zeroProb=options.zero_prob)
            )
            for motifName in options.motifNames 
        ]
    )
    sequenceSet = synthetic.GenerateSequenceNTimes(embedInBackground, options.numSeqs)
    synthetic.printSequences(outputFileName_core+".simdata", sequenceSet,
                             includeFasta=True, includeEmbeddings=True,
                             prefix=options.prefix)
   
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix")
    parser.add_argument("--pathToMotifs",
        default=simdna.ENCODE_MOTIFS_PATH)
    parser.add_argument("--bestHit", action="store_true")
    parser.add_argument("--motifNames", type=str, nargs='+', required=True)
    parser.add_argument("--max-motifs",type=int, required=True)
    parser.add_argument("--min-motifs",type=int, default=0)
    parser.add_argument("--mean-motifs",type=int, required=True)
    parser.add_argument("--zero-prob",type=float, required=False, default=0)
    parser.add_argument("--seqLength", type=int, required=True)
    parser.add_argument("--numSeqs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    options = parser.parse_args()
    do(options)
