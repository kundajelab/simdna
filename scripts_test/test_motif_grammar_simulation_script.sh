#!/usr/bin/env bash

#variable spacing
motifGrammarSimulation.py --prefix testprefix --motifName1 TAL1_known1 --motifName2 GATA_disc2 --seqLength 200 --numSeq 10 --generationSetting twoMotifsVariableSpacing --fixedSpacingOrMinSpacing 1 --maxSpacing 5

#variable spacing best hit
motifGrammarSimulation.py --motifName1 TAL1_known1 --motifName2 GATA_disc2 --bestHit --seqLength 200 --numSeq 100 --generationSetting twoMotifsVariableSpacing --fixedSpacingOrMinSpacing 1 --maxSpacing 5

#fixed spacing
motifGrammarSimulation.py --motifName1 TAL1_known1 --motifName2 GATA_disc2 --bestHit --seqLength 200 --numSeq 100 --generationSetting twoMotifsFixedSpacing --fixedSpacingOrMinSpacing 3

#two motifs
motifGrammarSimulation.py --motifName1 TAL1_known1 --motifName2 GATA_disc2 --bestHit --seqLength 200 --numSeq 100 --generationSetting twoMotifs

#all background
motifGrammarSimulation.py --motifName1 TAL1_known1 --motifName2 GATA_disc2 --bestHit --seqLength 200 --numSeq 100 --generationSetting allBackground

#singleMotif1
motifGrammarSimulation.py --motifName1 TAL1_known1 --motifName2 GATA_disc2 --bestHit --seqLength 200 --numSeq 100 --generationSetting singleMotif1

#singleMotif2
motifGrammarSimulation.py --motifName1 TAL1_known1 --motifName2 GATA_disc2 --bestHit --seqLength 200 --numSeq 100 --generationSetting singleMotif2
