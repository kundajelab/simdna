#!/usr/bin/env python
import os
import sys
import simdna
import simdna.util as util
import simdna.synthetic as synthetic
import simdna.pwm as pwm


def variableSpacingGrammar(options):
    pc = 0.001
    pathToMotifs = options.pathToMotifs
    loadedMotifs = synthetic.LoadedEncodeMotifs(pathToMotifs, pseudocountProb=pc)
    motifName1 = options.motifName1
    motifName2 = options.motifName2
    seqLength = options.seqLength
    numSeq = options.numSeq
    outputFileName = ("variableSpacingGrammarSimulation_motif1-"
                      +motifName1+"_motif2-"+motifName2
                      +"_seqLength"+str(seqLength)+"_numSeq"
                      +str(numSeq)+".simdata")

    kwargs={'loadedMotifs':loadedMotifs}
    theClass=synthetic.PwmSamplerFromLoadedMotifs
    motif1Generator=theClass(motifName=motifName1,**kwargs)
    motif2Generator=theClass(motifName=motifName2,**kwargs)
    motif1Embedder=synthetic.SubstringEmbedder(substringGenerator=motif1Generator)
    motif2Embedder=synthetic.SubstringEmbedder(substringGenerator=motif2Generator)

    embedders = []
    namePrefix="synthPos"
    separationGenerator=synthetic.MinMaxWrapper(
        synthetic.PoissonQuantityGenerator(options.meanSpacing),
        theMin=options.minSpacing,
        theMax=options.maxSpacing) 
    embedders.append(synthetic.EmbeddableEmbedder(
                        embeddableGenerator=synthetic.PairEmbeddableGenerator(
                            embeddableGenerator1=motif1Generator
                            ,embeddableGenerator2=motif2Generator
                            ,separationGenerator=separationGenerator
                        )
                    ))

    embedInBackground = synthetic.EmbedInABackground(
        backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength) 
        , embedders=embedders
        , namePrefix=namePrefix
    )

    sequenceSet = synthetic.GenerateSequenceNTimes(embedInBackground, numSeq)
    synthetic.printSequences(outputFileName, sequenceSet,
                             includeFasta=True, includeEmbeddings=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathToMotifs",
        default=simdna.ENCODE_MOTIFS_PATH)
    parser.add_argument("--motifName1", required=True)
    parser.add_argument("--motifName2", required=True)
    parser.add_argument("--seqLength", type=int, required=True)
    parser.add_argument("--numSeq", type=int, required=True)
    parser.add_argument("--minSpacing", type=int, required=True)
    parser.add_argument("--meanSpacing", type=float, required=True)
    parser.add_argument("--maxSpacing", type=int, required=True)
    
    options = parser.parse_args()
    variableSpacingGrammar(options) 
