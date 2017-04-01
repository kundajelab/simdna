from __future__ import absolute_import, division, print_function
from simdna.synthetic.embedders import AbstractEmbedder
from simdna.synthetic.core import AbstractSequenceSetGenerator
from simdna import util
from simdna.synthetic.core import EmbedInABackground
from simdna.synthetic.backgroundgen import ShuffledBackgroundGenerator
from simdna.synthetic.substringgen import (PwmSamplerFromLoadedMotifs,
                                           ReverseComplementWrapper)
from simdna.synthetic.embeddablegen import SubstringEmbeddableGenerator
from collections import OrderedDict

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
        #if (canEmbed == False):
            #print("Warning: trying to embed " + str(embeddable)
            #      + " at position " + str(self.startPos)
            #      + " which is already occupied")
        embeddable.embedInBackgroundStringArr(
         priorEmbeddedThings=priorEmbeddedThings,
         backgroundStringArr=backgroundStringArr,
         startPos=self.startPos)

    def getJsonableObject(self):
        """See superclass.
        """
        raise NotImplementedError()
