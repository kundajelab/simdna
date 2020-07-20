from __future__ import absolute_import, division, print_function
from simdna.synthetic.embedders import AbstractEmbedder
from simdna.synthetic.core import AbstractSequenceSetGenerator
from simdna.util import util
from simdna.synthetic.core import EmbedInABackground
from simdna.synthetic.backgroundgen import ShuffledBackgroundGenerator
from simdna.synthetic.substringgen import (PwmSamplerFromLoadedMotifs,
                                           ReverseComplementWrapper)
from simdna.synthetic.embeddablegen import SubstringEmbeddableGenerator
from collections import OrderedDict
from simdna import random


def parse_dnase_motif_embedder_string(embedderString, loadedMotifs):
    parseDnaseMotifEmbedderString(embedderString, loadedMotifs)


def parseDnaseMotifEmbedderString(embedderString, loadedMotifs):
    """Parse a string representing a motif and position

    Arguments:
        embedderString: of format <motif name>-<position in sequence>
        loadedMotifs: instance of :class:`.AbstractLoadedMotifs`

    Returns:
        An instance of :class:`FixedEmbeddableWithPosEmbedder`
    """
    motifName,pos = embedderString.split("-") 
    pwmSampler = PwmSamplerFromLoadedMotifs(
                    motifName=motifName,
                    loadedMotifs=loadedMotifs) 
    embeddableGenerator = SubstringEmbeddableGenerator(
                           substringGenerator=
                            ReverseComplementWrapper(pwmSampler))
    return FixedEmbeddableWithPosEmbedder(
            embeddableGenerator=embeddableGenerator,
            startPos=int(pos))


class DnaseSimulation(AbstractSequenceSetGenerator):
    """Simulation based on a file that details the sequences (which may be
    shuffled) and the motifs+positions in the sequences

    Arguments:
        dnaseSimulationFile: file with a title, and columns:
            sequenceName<tab>sequence<tab>motif1-pos1,motif2-pos2...
        loadedMotifs: instance of :class:`.AbstractLoadedMotifs`
        shuffler: instance of :class:`.AbstractShuffler`
    """

    def __init__(self, dnaseSimulationFile, loadedMotifs, shuffler):
        self.dnaseSimulationFile = dnaseSimulationFile
        self.loadedMotifs = loadedMotifs
        self.shuffler=shuffler

    def generateSequences(self):
        fileHandle = util.get_file_handle(self.dnaseSimulationFile)
        for lineNumber, line in enumerate(fileHandle):
            if (lineNumber > 0): #ignore title
                inp = util.default_tab_seppd(line)
                sequenceName = inp[0]
                backgroundGenerator = ShuffledBackgroundGenerator(
                            string=inp[1], shuffler=self.shuffler)
                embedders = [parseDnaseMotifEmbedderString(
                              embedderString, self.loadedMotifs)
                             for embedderString in inp[2].split(",")
                             if len(embedderString) > 0]
                yield SingleDnaseSequenceGenerator(
                    backgroundGenerator=backgroundGenerator,
                    dnaseMotifEmbedders=embedders,
                    sequenceName=sequenceName).generateSequence()

    def getJsonableObject(self):
        """See superclass 
        """
        return OrderedDict(
            [('dnaseSimulationFile', self.dnaseSimulationFile),
             ('shuffler', self.shuffler.getJsonableObject())]) 


class SingleDnaseSequenceGenerator(object):
    def __init__(self, backgroundGenerator, dnaseMotifEmbedders, sequenceName):
        self.backgroundGenerator = backgroundGenerator 
        self.dnaseMotifEmbedders = dnaseMotifEmbedders
        self.sequenceName = sequenceName

    def generate_sequence(self):
        self.generateSequence()

    def generateSequence(self):
        return EmbedInABackground.\
         generateSequenceGivenBackgroundGeneratorAndEmbedders(
            backgroundGenerator=self.backgroundGenerator,
            embedders=self.dnaseMotifEmbedders,
            sequenceName=self.sequenceName) 


class FixedEmbeddableWithPosEmbedder(AbstractEmbedder):
    """Embeds a given :class:`.AbstractEmbeddable` at a given pos.

    Embeds a given instance of :class:`.AbstractEmbeddable` within the
    background sequence, at a given position. Could result in overlapping
    embeddings if positions are too close.

    Arguments:
        embeddable: instance of :class:`.AbstractEmbeddable`
        startPos: an int
    """

    def __init__(self, embeddableGenerator, startPos):
        self.embeddableGenerator = embeddableGenerator
        self.startPos = startPos
        super(FixedEmbeddableWithPosEmbedder, self).__init__(None)

    def _embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo):
        """Shoves the designated embeddable at the designated position
        Skips if some of the positions are already occupied.
        """
        embeddable = self.embeddableGenerator.generateEmbeddable()
        canEmbed = embeddable.canEmbed(priorEmbeddedThings, self.startPos)
        
        #randomly pick a value for searchLeft
        if random.random() < 0.5:
            searchLeft=True
        else:
            searchLeft=False 

        validEmbeddingPos = self._getValidEmbeddingPos(
                                embeddable=embeddable,
                                priorEmbeddedThings=priorEmbeddedThings,
                                backgroundStringArr=backgroundStringArr,
                                startingPosToSearchFrom=self.startPos,
                                searchLeft=searchLeft)
        #if couldn't find a valid pos, search in the other direction
        if (validEmbeddingPos is None):
            validEmbeddingPos = self._getValidEmbeddingPos(
                                    embeddable=embeddable,
                                    priorEmbeddedThings=priorEmbeddedThings,
                                    backgroundStringArr=backgroundStringArr,
                                    startingPosToSearchFrom=self.startPos,
                                    searchLeft=(searchLeft==False))
        if (validEmbeddingPos is None):
            print("Warning: could not find a place to embed "+str(embeddable)
                  +"; bailing")
            return
        else:
            embeddable.embedInBackgroundStringArr(
             priorEmbeddedThings=priorEmbeddedThings,
             backgroundStringArr=backgroundStringArr,
             startPos=validEmbeddingPos)

    def _getValidEmbeddingPos(self, embeddable,
                                    priorEmbeddedThings,
                                    backgroundStringArr,
                                    startingPosToSearchFrom,
                                    searchLeft):
        posToQuery = startingPosToSearchFrom 
        maxLen = len(backgroundStringArr)
        embeddableLen = len(embeddable)
        #search left/right (according to the value of searchLeft) for
        #a valid position at which to embed the embeddable
        while (posToQuery > 0 and posToQuery < maxLen):
            canEmbed = embeddable.canEmbed(priorEmbeddedThings, posToQuery) 
            if (canEmbed):
                return posToQuery
            if (searchLeft):
                posToQuery -= 1 
            else:
                posToQuery += 1
        return None 

    def getJsonableObject(self):
        """See superclass.
        """
        raise NotImplementedError()
