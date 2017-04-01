#!/usr/bin/env bash

#standard density motif simulation
densityMotifSimulation.py --prefix gata --motifNames GATA_disc1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 10

#standard density motif simulation, best hit
densityMotifSimulation.py --prefix gata --motifNames GATA_disc1 --bestHit --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 10

#standard density motif simulation, with zero-inflation of the poisson distribution so ~50% have no motifs
densityMotifSimulation.py --prefix gata --motifNames GATA_disc1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 10 --zero-prob 0.5
