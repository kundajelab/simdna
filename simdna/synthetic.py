from __future__ import absolute_import, division, print_function
from . import util, pwm, fileProcessing as fp
import argparse
import numpy as np
from simdna import random
import math
from collections import OrderedDict
import json
import re
import itertools
from . import dinuc_shuffle


class LabelGenerator(object):
    """Generate labels for a generated sequence.

    Arguments:
        labelNames: an array of strings that are the names of the labels
        labelsFromGeneratedSequenceFunction: function that accepts
            an instance of :class:`.GeneratedSequence` and returns an array
            of the labels (eg: an array of ones and zeros indicating if
            the criteria for various labels are met)
    """
    def __init__(self, labelNames, labelsFromGeneratedSequenceFunction):
        self.labelNames = labelNames
        self.labelsFromGeneratedSequenceFunction = labelsFromGeneratedSequenceFunction

    def generateLabels(self, generatedSequence):
        """calls self.labelsFromGeneratedSequenceFunction.

        Arguments:
            generatedSequence: an instance of :class:`.GeneratedSequence`
        """
        return self.labelsFromGeneratedSequenceFunction(
            self, generatedSequence)


class IsInTraceLabelGenerator(LabelGenerator):
    """LabelGenerator where labels match which embedders are called.

    A special kind of LabelGenerator where the names of the labels
        are the names of embedders, and the label is 1 if a particular
        embedder has been called on the sequence and 0 otherwise.
    """ 
    def __init__(self, labelNames):
        def labelsFromGeneratedSequenceFunction(self, generatedSequence):
            return [(1 if generatedSequence.additionalInfo.isInTrace(x) else 0)
                    for x in self.labelNames]
        super(IsInTraceLabelGenerator, self).__init__(
            labelNames, labelsFromGeneratedSequenceFunction)


def printSequences(outputFileName, sequenceSetGenerator,
                   includeEmbeddings=False, labelGenerator=None,
                   includeFasta=False, prefix=None):
    """Print a series of synthetic sequences.

    Given an output filename, and an instance of
        :class:`.AbstractSequenceSetGenerator`, will call the
        sequenceSetGenerator and print the generated sequences
        to the output file. Will also create a file "info_outputFileName.txt"
        in the same directory as outputFileName that contains
        all the information about sequenceSetGenerator.

    Arguments:
        outputFileName: string

        sequenceSetGenerator: instance of
            :class:`.AbstractSequenceSetGenerator`
    
        includeEmbeddings: a boolean indicating whether to print a
            column that lists the embeddings
    
        labelGenerator: optional instance of :class:`.LabelGenerator`

        includeFasta: optional boolean indicating whether to also
            print out the generated sequences in fasta format
            (the file will be produced with a .fa extension)

        prefix: string - this will be prefixed in front of the generated
            sequence ids, followed by a hyphen
    """
    ofh = fp.getFileHandle(outputFileName, 'w')
    if (includeFasta):
        fastaOfh = fp.getFileHandle(fp.getFileNameParts(
            outputFileName).getFilePathWithTransformation(
            lambda x: x, extension=".fa"), 'w')
    ofh.write("seqName\tsequence"
              + ("\tembeddings" if includeEmbeddings else "")
              + ("\t" +
                 "\t".join(labelGenerator.labelNames)
                 if labelGenerator is not None else "") + "\n")
    generatedSequences = sequenceSetGenerator.generateSequences()  # returns a generator
    for generatedSequence in generatedSequences:
        ofh.write((prefix+"-" if prefix is not None else "")
                  + generatedSequence.seqName + "\t" + generatedSequence.seq
                  + ("\t" + ",".join(str(x)
                     for x in generatedSequence.embeddings)
                         if includeEmbeddings else "")
                  + ("\t" + "\t".join(str(x) for x in labelGenerator.generateLabels(
                      generatedSequence)) if labelGenerator is not None else "")
                  + "\n")
        if (includeFasta):
            fastaOfh.write(">" + (prefix+"-" if prefix is not None else "")
                               + generatedSequence.seqName + "\n")
            fastaOfh.write(generatedSequence.seq + "\n")

    ofh.close()
    if (includeFasta):
        fastaOfh.close()
    infoFilePath = fp.getFileNameParts(outputFileName).getFilePathWithTransformation(
        lambda x: x + "_info", extension=".txt")

    ofh = fp.getFileHandle(infoFilePath, 'w')
    ofh.write(util.formattedJsonDump(sequenceSetGenerator.getJsonableObject()))
    ofh.close()


def read_simdata_file(simdata_file, one_hot_encode=False, ids_to_load=None):
    ids = []
    sequences = []
    embeddings = []
    labels = []
    if (ids_to_load is not None):
        ids_to_load = set(ids_to_load)
    def action(inp, line_number):
        if (line_number > 1):
            if (ids_to_load is None or (inp[0] in ids_to_load)):
                ids.append(inp[0]) 
                sequences.append(inp[1])
                embeddings.append(getEmbeddingsFromString(inp[2]))
                labels.append([int(x) for x in inp[3:]])
    fp.performActionOnEachLineOfFile(
        fileHandle=fp.getFileHandle(simdata_file),
        action=action,
        transformation=fp.defaultTabSeppd)
    return util.enum(
            ids=ids,
            sequences=sequences,
            embeddings=embeddings,
            labels=np.array(labels))


class DefaultNameMixin(object):
    """Basic functionality for classes that have a self.name attribute.
    
    The self.name attribute is typically used to leave a trace in
    an instance of :class:`.AdditionalInfo`
    
    Arguments:
        name: string
    """
    def __init__(self, name):
        if (name == None):
            name = self.getDefaultName()
        self.name = name

    def getDefaultName(self):
        return type(self).__name__


class AbstractPositionGenerator(DefaultNameMixin):
    """Generate a start position at which to embed something

    Given the length of the background sequence and the length
    of the substring you are trying to embed, will return a start position
    to embed the substring at.
    """

    def generatePos(self, lenBackground, lenSubstring, additionalInfo=None):
        """Generate the position to embed in.

        Arguments:
            lenBackground: int, length of background sequence

            lenSubstring: int, lenght of substring to embed

            additionalInfo: optional, instance of :class:`.AdditionalInfo`. Is
                used to leave a trace that this positionGenerator was called

        Returns:
            An integer which is the start index to embed in.
        """
        if (additionalInfo is not None):
            additionalInfo.updateTrace(self.name)
        return self._generatePos(lenBackground, lenSubstring, additionalInfo)

    def _generatePos(self, lenBackground, lenSubstring, additionalInfo):
        """Generate the position to embed in - this method should be
        overriden by the subclass. See
        :func:`.AbstractPositionGenerator.generatePos` for documentaiton
        on the arguments.
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


class UniformPositionGenerator(AbstractPositionGenerator):
    """Sample position uniformly at random.
    
    Samples a start position to embed the substring in uniformly at random;
        does not return positions that are too close to the end of the
        background sequence to embed the full substring.

    Arguments:
        name: string, see :class:`.DefaultNameMixin`
    """

    def __init__(self, name=None):
        super(UniformPositionGenerator, self).__init__(name)

    def _generatePos(self, lenBackground, lenSubstring, additionalInfo):
        return sampleIndexWithinRegionOfLength(lenBackground, lenSubstring)

    def getJsonableObject(self):
        """See superclass.
        """
        return "uniform"

#instantiate a UniformPositionGenerator for general use
uniformPositionGenerator = UniformPositionGenerator()


class InsideCentralBp(AbstractPositionGenerator):
    """For embedding within only the central region of a background.
        
    Returns a position within the central region of a background
        sequence, sampled uniformly at random

    Arguments:
        centralBp: int, the number of bp, centered in the
            middle of the background, from which to sample the position.
            Is NOT +/- centralBp around the middle
            (is +/- centralBp/2 around the middle). If the background
            sequence is even and centralBp is odd, the shorter region
            will go on the left.
        name: string - see :class:`.DefaultNameMixin`
    """

    def __init__(self, centralBp, name=None):
        """
        """
        self.centralBp = centralBp
        super(InsideCentralBp, self).__init__(name)

    def _generatePos(self, lenBackground, lenSubstring, additionalInfo):
        if (lenBackground < self.centralBp):
            raise RuntimeError("The background length should be atleast as long as self.centralBp; is " +
                               str(lenBackground) + " and " + str(self.centralBp) + " respectively")
        startIndexForRegionToEmbedIn = int(
            lenBackground / 2) - int(self.centralBp / 2)
        indexToSample = startIndexForRegionToEmbedIn + \
            sampleIndexWithinRegionOfLength(self.centralBp, lenSubstring)
        return int(indexToSample)

    def getJsonableObject(self):
        """See superclass.
        """
        return "insideCentral-" + str(self.centralBp)


class OutsideCentralBp(AbstractPositionGenerator):
    """For embedding only OUTSIDE a central region of a background seq.

    Returns a position OUTSIDE the central region of a background sequence,
        sampled uniformly at random. Complement of InsideCentralBp.

    Arguments:
        centralBp: int, the centralBp to avoid embedding in. See the docs
            for :class:`.InsideCentralBp` for more details (this is the
            complement).
    """

    def __init__(self, centralBp, name=None):
        self.centralBp = centralBp
        super(OutsideCentralBp, self).__init__(name)

    def _generatePos(self, lenBackground, lenSubstring, additionalInfo):
        # choose whether to embed in the left or the right
        if random.random() > 0.5:
            left = True
        else:
            left = False
        # embeddableLength is the length of the region we are considering
        # embedding in
        embeddableLength = 0.5 * (lenBackground - self.centralBp)
        # if lenBackground-self.centralBp is odd, the longer region
        # goes on the left (inverse of the shorter embeddable region going on the left in
        # the centralBpToEmbedIn case
        if (left):
            embeddableLength = math.ceil(embeddableLength)
            startIndexForRegionToEmbedIn = 0
        else:
            embeddableLength = math.floor(embeddableLength)
            startIndexForRegionToEmbedIn = math.ceil(
                (lenBackground - self.centralBp) / 2) + self.centralBp
        indexToSample = startIndexForRegionToEmbedIn + \
            sampleIndexWithinRegionOfLength(embeddableLength, lenSubstring)
        return int(indexToSample)

    def getJsonableObject(self):
        """See superclass.
        """
        return "outsideCentral-" + str(self.centralBp)


class GeneratedSequence(object):
    """An object representing a sequence that has been generated.

    Arguments:
        seqName: string representing the name/id of the sequence

        seq: string representing the final generated sequence

        embeddings: an array of :class:`.Embedding` objects.

        additionalInfo: an instance of :class:`.AdditionalInfo`
    """

    def __init__(self, seqName, seq, embeddings, additionalInfo):
        self.seqName = seqName
        self.seq = seq
        self.embeddings = embeddings
        self.additionalInfo = additionalInfo


class Embedding(object):
    """Represents something that has been embedded in a sequence.

    Think of this as a combination of an embeddable + a start position.

    Arguments:
        what: object representing the thing that has been embedded.\
            Should have`` __str__`` and ``__len__`` defined.\
            Often is an instance of :class:`.AbstractEmbeddable`

        startPos: int, the position relative to the start of the parent\
            sequence at which seq has been embedded
    """

    def __init__(self, what, startPos):
        self.what = what
        self.startPos = startPos

    def __str__(self):
        return "pos-" + str(self.startPos) + "_" + str(self.what)

    @classmethod
    def fromString(cls, string, whatClass=None):
        """Recreate an :class:`.Embedding` object from a string.

        Arguments:
            string: assumed to have format:\
                ``description[-|_]startPos[-|_]whatString``, where
                ``whatString`` will be provided to ``whatClass``

            whatClass: the class (usually a :class:`.AbstractEmbeddable`) that\
                will be used to instantiate the what from the whatString

        Returns:
            The Embedding class called with
            ``what=whatClass.fromString(whatString)`` and
            ``startPos=int(startPos)``
        """
        if (whatClass is None):
            whatClass = StringEmbeddable
        # was printed out as pos-[startPos]_[what], but the
        #[what] may contain underscores, hence the maxsplit
        # to avoid splitting on them.
        p = re.compile(r"pos\-(\d+)_(.*)$")
        m = p.search(string)
        startPos = m.group(1)
        whatString = m.group(2) 
        return cls(what=whatClass.fromString(whatString),
                   startPos=int(startPos))


def getEmbeddingsFromString(string):
    """Get a series of :class:`.Embedding` objects from a string.
    
    Splits the string on commas, and then passes the comma-separated vals
        to :func:`.Embedding.fromString`

    Arguments:
        string: The string to turn into an array of Embedding objects

    Returns:
        an array of :class:`.Embedding` objects
    """
    if len(string) == 0:
        return []
    else:
        embeddingStrings = string.split(",")
        return [Embedding.fromString(x) for x in embeddingStrings]


class AbstractSequenceSetGenerator(object):
    """A generator for a collection of generated sequences.
    """

    def generateSequences(self):
        """The generator; implementation should have a yield.

        Called as
        ``generatedSequences = sequenceSetGenerator.generateSequences()``

        ``generateSequences`` can then be iterated over.

        Returns:
            A generator of GeneratedSequence objects
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


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
        fileHandle = fp.getFileHandle(self.dnaseSimulationFile)
        for lineNumber, line in enumerate(fileHandle):
            if (lineNumber > 0): #ignore title
                inp = fp.defaultTabSeppd(line)
                sequenceName = inp[0]
                backgroundGenerator = ShuffledBackgroundGenerator(
                            string=inp[1], shuffler=self.shuffler)
                embedders = [parseDnaseMotifEmbedderString(
                              embedderString, self.loadedMotifs)
                             for embedderString in inp[2].split(",")]
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


class ChainSequenceSetGenerators(AbstractSequenceSetGenerator):
    """Chains several generators together.

    Arguments:
        generators: instances of :class:`.AbstractSequenceSetGenerator`.
    """

    def __init__(self, *generators):
        self.generators = generators

    def generateSequences(self):
        """A chain of generators

        Returns:
            A chain of generators
        """
        for item in itertools.chain(*[generator.generateSequences()
                                      for generator in self.generators]):
            yield item

    def getJsonableObject(self):
        """See superclass 
        """
        return OrderedDict([('generators',
                             [x.getJsonableObject() for x
                             in self.generators])]) 


class GenerateSequenceNTimes(AbstractSequenceSetGenerator):
    """Call a :class:`.AbstractSingleSequenceGenerator` N times.
            
    Arguments:
        singleSetGenerator: an instance of
            :class:`.AbstractSequenceSetGenerator`
        N: integer, the number of times to call singleSetGenerator
    """

    def __init__(self, singleSetGenerator, N):
        self.singleSetGenerator = singleSetGenerator
        self.N = N

    def generateSequences(self):
        """A generator that calls self.singleSetGenerator N times.

        Returns:
            a generator that will call self.singleSetGenerator N times. 
        """
        for i in range(self.N):
            yield self.singleSetGenerator.generateSequence()

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("numSeq", self.N), ("singleSetGenerator", self.singleSetGenerator.getJsonableObject())])


class AbstractSingleSequenceGenerator(object):
    """Generate a single sequence.

    Arguments:
        namePrefix: the GeneratedSequence object has a field
            for the object's name; this is the prefix associated
            with that name. The suffix is the value of a counter that
            is incremented every time
    """

    def __init__(self, namePrefix=None):
        self.namePrefix = namePrefix if namePrefix is not None else "synth"

    def generateSequence(self):
        """Generate the sequence.

        Returns:
            An instance of :class:`.GeneratedSequence`
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


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


class EmbedInABackground(AbstractSingleSequenceGenerator):
    """Generate a background sequence and embed smaller sequences in it.
    
    Takes a backgroundGenerator and a series of embedders. Will
    generate the background and then call each of the embedders in
    succession. Then returns the result.

    Arguments:
        backgroundGenerator: instance of
            :class:`.AbstractBackgroundGenerator`
        embedders: array of instances of :class:`.AbstractEmbedder`
        namePrefix: see parent
    """

    def __init__(self, backgroundGenerator, embedders, namePrefix=None):
        super(EmbedInABackground, self).__init__(namePrefix)
        self.backgroundGenerator = backgroundGenerator
        self.embedders = embedders
        self.sequenceCounter = 0

    @staticmethod
    def generateSequenceGivenBackgroundGeneratorAndEmbedders(
        backgroundGenerator, embedders, sequenceName):
        additionalInfo = AdditionalInfo()
        backgroundString = backgroundGenerator.generateBackground()
        backgroundStringArr = [x for x in backgroundString]
        # priorEmbeddedThings keeps track of what has already been embedded
        priorEmbeddedThings = PriorEmbeddedThings_numpyArrayBacked(
            len(backgroundStringArr))
        for embedder in embedders:
            embedder.embed(backgroundStringArr,
                           priorEmbeddedThings, additionalInfo)
        return GeneratedSequence(sequenceName,
                                 "".join(backgroundStringArr),
                                 priorEmbeddedThings.getEmbeddings(),
                                 additionalInfo)
 
    def generateSequence(self):
        """Produce the sequence.
        
        Generates a background using self.backgroundGenerator,
        splits it into an array, and passes it to each of
        self.embedders in turn for embedding things.

        Returns:
            An instance of :class:`.GeneratedSequence`
        """
        toReturn = EmbedInABackground.\
         generateSequenceGivenBackgroundGeneratorAndEmbedders(
            backgroundGenerator=self.backgroundGenerator,
            embedders=self.embedders,
            sequenceName=self.namePrefix + str(self.sequenceCounter))
        self.sequenceCounter += 1
        return toReturn

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "EmbedInABackground"),
                            ("namePrefix", self.namePrefix),
                            ("backgroundGenerator",
                             self.backgroundGenerator.getJsonableObject()),
                            ("embedders",
                             [x.getJsonableObject() for x in self.embedders])
                            ])


class AdditionalInfo(object):
    """Used to keep track of which embedders/ops were
    called and how many times.

    An instance of AdditionalInfo is meant to be an attribute of
        a :class:`.GeneratedSequence` object. It keeps track of things
        like embedders, position generators, etc.

    Has self.trace which is a dictionary from operatorName->int
        and which records operations that were called in the
        process of embedding things in the sequence. At the time
        of writing, operatorName is typically just the name of the
        embedder.
    """
    def __init__(self):
        self.trace = OrderedDict()  # a trace of everything that was called.
        self.additionalInfo = OrderedDict()  # for more ad-hoc messages

    def isInTrace(self, operatorName):
        """Return True if operatorName has been called on the sequence.
        """
        return operatorName in self.trace

    def updateTrace(self, operatorName):
        """Increment count for the number of times operatorName was called.
        """
        if (operatorName not in self.trace):
            self.trace[operatorName] = 0
        self.trace[operatorName] += 1

    def updateAdditionalInfo(self, operatorName, value):
        """Can be used to store any additional information on operatorName.
        """
        self.additionaInfo[operatorName] = value


class AbstractPriorEmbeddedThings(object):
    """Keeps track of what has already been embedded in a sequence.
    """

    def canEmbed(self, startPos, endPos):
        """Test whether startPos-endPos is available for embedding.

        Arguments:
            startPos: int, starting index
            endPos: int, ending index+1 (same semantics as array-slicing)

        Returns:
            True if startPos:endPos is available for embedding
        """
        raise NotImplementedError()

    def addEmbedding(self, startPos, what):
        """Records the embedding of a :class:`AbstractEmbeddable`.

        Embeds ``what`` from ``startPos`` to ``startPos+len(what)``.
        Creates an :class:`Embedding` object.

        Arguments:
            startPos: int, the starting position at which to embed.
            what: instance of :class:`AbstractEmbeddable`
        """
        raise NotImplementedError()

    def getNumOccupiedPos(self):
        """
        Returns:
            Number of posiitons that are filled with some kind of embedding
        """
        raise NotImplementedError()

    def getTotalPos(self):
        """
        Returns:
            Total number of positions (occupied and unoccupoed) available
        to embed things in.
        """
        raise NotImplementedError()

    def getEmbeddings(self):
        """
        Returns:
            A collection of Embedding objects
        """
        raise NotImplementedError()


class PriorEmbeddedThings_numpyArrayBacked(AbstractPriorEmbeddedThings):
    """A numpy-array based implementation of
    :class:`.AbstractPriorEmbeddedThings`.

    Uses a numpy array where positions are set to 1 if they are occupied,
    to determine which positions are occupied and which are not.
    See superclass for more documentation.

    Arguments:
        seqLen: integer indicating length of the sequence you are embedding in
    """

    def __init__(self, seqLen):
        self.seqLen = seqLen
        self.arr = np.zeros(seqLen)
        self.embeddings = []

    def canEmbed(self, startPos, endPos):
        """See superclass.
        """
        return np.sum(self.arr[startPos:endPos]) == 0

    def addEmbedding(self, startPos, what):
        """See superclass.
        """
        self.arr[startPos:startPos + len(what)] = 1
        self.embeddings.append(Embedding(what=what, startPos=startPos))

    def getNumOccupiedPos(self):
        """See superclass.
        """
        return np.sum(self.arr)

    def getTotalPos(self):
        """See superclass.
        """
        return len(self.arr)

    def getEmbeddings(self):
        """See superclass.
        """
        return self.embeddings


class AbstractEmbeddable(object):
    """Represents a thing which can be embedded.

        An :class:`.AbstractEmbeddable` + a position = an :class:`.Embedding`
    """

    def __len__(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def getDescription(self):
        """Return a concise description of the embeddable.

        This should be concise and shouldn't contain spaces. It will often
        be used when generating the __str__ representation of the embedabled.
        """
        raise NotImplementedError()

    def canEmbed(self, priorEmbeddedThings, startPos):
        """Checks whether embedding is possible at a given pos.

        Accepts an instance of :class:`AbstractPriorEmbeddedThings` and
        a ``startPos``, and checks if ``startPos`` is viable given the
        contents of ``priorEmbeddedThings``.

        Arguments:
            priorEmbeddedThings: instance of
        :class:`AbstractPriorEmbeddedThings`

            startPos: int; the position you are considering embedding self at

        Returns:
            A boolean indicating whether self can be embedded at startPos,
        given the things that have already been embedded.
        """
        raise NotImplementedError()

    def embedInBackgroundStringArr(self, priorEmbeddedThings, backgroundStringArr, startPos):
        """Embed self in a background string.

        Will embed self at ``startPos`` in ``backgroundStringArr``,
        and will update ``priorEmbeddedThings`` accordingly.

        Arguments:
            priorEmbeddedThings: instance of
        :class:`AbstractPriorEmbeddedThings`
            backgroundStringArr: an array of characters representing
        the background
            startPos: integer; the position to embed self at
        """
        raise NotImplementedError()

    @classmethod
    def fromString(cls, theString):
        """Generate an instance of the embeddable from the provided string.
        """
        raise NotImplementedError()


class StringEmbeddable(AbstractEmbeddable):
    """A string that is to be embedded in a background.

    Represents a string (such as a sampling from a pwm) that is to
    be embedded in a background. See docs for superclass.

    Arguments:
        string: the core string to be embedded

        stringDescription: a short descriptor prefixed before the\
        ``__str__`` representation of the embeddable.\
        Should not contain a hyphen. Defaults to "".
    """

    def __init__(self, string, stringDescription=""):
        self.string = string
        self.stringDescription = stringDescription

    def __len__(self):
        return len(self.string)

    def __str__(self):
        return self.stringDescription + ("-" if self.stringDescription != "" else "") + self.string

    def getDescription(self):
        """See superclass.
        """
        return self.stringDescription

    def canEmbed(self, priorEmbeddedThings, startPos):
        """See superclass.
        """
        return priorEmbeddedThings.canEmbed(startPos, startPos + len(self.string))

    def embedInBackgroundStringArr(self, priorEmbeddedThings,
                                         backgroundStringArr, startPos):
        """See superclass.
        """
        positions_left = len(backgroundStringArr)-startPos
        if (positions_left < len(self.string)):
            print("Warning: length of background is "
                  +str(len(backgroundStringArr))
                  +" but was asked to embed string of length "
                  +str(len(self.string))+" at position "
                  +str(startPos)+"; truncating")
            string_to_embed = self.string[:positions_left]
        else:
            string_to_embed = self.string
        backgroundStringArr[startPos:
         startPos+len(string_to_embed)] = string_to_embed
        priorEmbeddedThings.addEmbedding(startPos, self)

    @classmethod
    def fromString(cls, theString):
        """Generates a StringEmbeddable from the provided string.

        Arguments:
            theString: string of the format ``stringDescription-coreString``.\
        Will then return:\
      ``StringEmbeddable(string=coreString, stringDescription=stringDescription)``

        Returns:
            An instance of :class:`.StringEmbeddable`
        """
        if ("-" in theString):
            p = re.compile(r"((revComp\-)?(.*))\-(.*)$")
            m = p.search(theString)
            stringDescription = m.group(1)
            coreString = m.group(4)
            return cls(string=coreString, stringDescription=stringDescription)
        else:
            return cls(string=theString)


class PairEmbeddable(AbstractEmbeddable):
    """Embed two embeddables with some separation.

    Arguments:
        embeddable1: instance of :class:`.AbstractEmbeddable`.\
        First embeddable to be embedded. If a string is provided, will\
        be wrapped in :class:`.StringEmbeddable`

        embeddable2: second embeddable to be embedded. Type information\
        similar to that of ``embeddable1``

        separation: int of distance separating embeddable1 and embeddable2

        embeddableDescription: a concise descriptive string prefixed in\
        front when generating a __str__ representation of the embeddable.\
        Should not contain a hyphen.

        nothingInBetween: if true, then nothing else is allowed to be\
        embedded in the gap between embeddable1 and embeddable2.
    """

    def __init__(self, embeddable1, embeddable2, separation,
                       embeddableDescription="", nothingInBetween=True):
        if (isinstance(embeddable1, str)):
            embeddable1 = StringEmbeddable(string=embeddable1)
        if (isinstance(embeddable2, str)):
            embeddable2 = StringEmbeddable(string=embeddable2)
        self.embeddable1 = embeddable1
        self.embeddable2 = embeddable2
        self.separation = separation
        self.embeddableDescription = embeddableDescription
        self.nothingInBetween = nothingInBetween

    def __len__(self):
        return len(self.embeddable1) + self.separation + len(self.embeddable2)

    def __str__(self):
        return self.embeddableDescription +\
               ("-" if self.embeddableDescription != "" else "") +\
               str(self.embeddable1) + "-Gap" + str(self.separation) +\
               "-" + str(self.embeddable2)

    def getDescription(self):
        """See superclass.
        """
        return self.embeddableDescription

    def canEmbed(self, priorEmbeddedThings, startPos):
        """See superclass.
        """
        if (self.nothingInBetween):
            return priorEmbeddedThings.canEmbed(startPos, startPos + len(self))
        else:
            return (priorEmbeddedThings.canEmbed(startPos, startPos + len(self.embeddable1))
                    and priorEmbeddedThings.canEmbed(startPos + len(self.embeddable1) + self.separation, startPos + len(self)))

    def embedInBackgroundStringArr(self, priorEmbeddedThings,
                                         backgroundStringArr, startPos):
        """See superclass.

        If ``self.nothingInBetween``, then all the intervening positions
        between the two embeddables will be marked as occupied. Otherwise,
        only the positions occupied by the embeddables will be marked
        as occupied.
        """
        self.embeddable1.embedInBackgroundStringArr(
            priorEmbeddedThings, backgroundStringArr, startPos)
        self.embeddable2.embedInBackgroundStringArr(
            priorEmbeddedThings, backgroundStringArr,
            startPos+len(self.embeddable1)+self.separation)
        if (self.nothingInBetween):
            priorEmbeddedThings.addEmbedding(startPos, self)
        else:
            priorEmbeddedThings.addEmbedding(startPos, self.embeddable1)
            priorEmbeddedThings.addEmbedding(
                startPos + len(self.embeddable1) + self.separation, self.embeddable2)


class AbstractEmbedder(DefaultNameMixin):
    """Produces :class:`AbstractEmbeddable` objects and
    embeds them in a sequence.
    """

    def embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo=None):
        """Embeds things in the provided ``backgroundStringArr``.

        Modifies backgroundStringArr to include whatever has been embedded.

        Arguments:
            backgroundStringArr: array of characters\
        representing the background string

            priorEmbeddedThings: instance of\
        :class:`.AbstractPriorEmbeddedThings`

            additionalInfo: instance of :class:`.AdditionalInfo`;\
        allows the embedder to send back info about what it did

        Returns:
            The modifed ``backgroundStringArr``
        """
        if (additionalInfo is not None):
            additionalInfo.updateTrace(self.name)
        return self._embed(backgroundStringArr, priorEmbeddedThings, additionalInfo)

    def _embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo):
        """The actual implementation of _embed to be overridden by
        the subclass.

        See docs for :func:`.AbstractEmbedder.embed`
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


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


class EmbeddableEmbedder(AbstractEmbedder):
    """Embeds an instance of :class:`.AbstractEmbeddable` at a sampled pos.

    Embeds instances of :class:`.AbstractEmbeddable` within the
    background sequence, at a position sampled from a distribution.
    Only embeds at unoccupied positions.

    Arguments:
        embeddableGenerator: instance of :class:`.AbstractEmbeddableGenerator`

        positionGenerator: instance of :class:`.AbstractPositionGenerator`
    """

    def __init__(self, embeddableGenerator,
                       positionGenerator=uniformPositionGenerator, name=None):
        self.embeddableGenerator = embeddableGenerator
        self.positionGenerator = positionGenerator
        super(EmbeddableEmbedder, self).__init__(name)

    def _embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo):
        """See superclass.

        Calls self.embeddableGenerator to determine the
        embeddable to embed. Then calls self.positionGenerator to
        determine the start position at which to embed it.
        If the position is occupied, will resample from
        ``self.positionGenerator``. Will warn if tries to
        resample too many times.
        """
        embeddable = self.embeddableGenerator.generateEmbeddable()
        canEmbed = False
        tries = 0
        while canEmbed == False:
            tries += 1
            startPos = self.positionGenerator.generatePos(
                len(backgroundStringArr), len(embeddable), additionalInfo)
            canEmbed = embeddable.canEmbed(priorEmbeddedThings, startPos)
            if (tries % 10 == 0): 
                print("Warning: made " + str(tries) + " attemps at trying to embed " + str(embeddable) + " in region of length " + str(
                    priorEmbeddedThings.getTotalPos()) + " with " + str(priorEmbeddedThings.getNumOccupiedPos()) + " occupied sites")
        embeddable.embedInBackgroundStringArr(
            priorEmbeddedThings, backgroundStringArr, startPos)

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("embeddableGenerator", self.embeddableGenerator.getJsonableObject()), ("positionGenerator", self.positionGenerator.getJsonableObject())])


class XOREmbedder(AbstractEmbedder):
    """Calls exactly one of the supplied embedders.

    Arguments:
        embedder1: instance of :class:`.AbstractEmbedder`

        embedder2: instance of :class:`.AbstractEmbedder`

        probOfFirst: probability of calling the first embedder
    """

    def __init__(self, embedder1, embedder2, probOfFirst, name=None):
        self.embedder1 = embedder1
        self.embedder2 = embedder2
        self.probOfFirst = probOfFirst
        super(XOREmbedder, self).__init__(name)

    def _embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo):
        """See superclass.
        """
        if (random.random() < self.probOfFirst):
            embedder = self.embedder1
        else:
            embedder = self.embedder2
        return embedder.embed(backgroundStringArr, priorEmbeddedThings, additionalInfo)

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "XOREmbedder"), ("embedder1", self.embedder1.getJsonableObject()), ("embedder2", self.embedder2.getJsonableObject()), ("probOfFirst", self.probOfFirst)])


class AllEmbedders(AbstractEmbedder):
    """Wrapper around a list of embedders that calls each one in turn.

    Useful to nest under a :class:`.RandomSubsetOfEmbedders`

    Arguments:
        embedders: an iterable of :class:`.AbstractEmbedder` objects.
    """
    def __init__(self, embedders, name=None):
        self.embedders = embedders
        super(AllEmbedders, self).__init__(name)

    def _embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo):
        """See superclass.
        """
        for embedder in self.embedders:
            embedder.embed(backgroundStringArr,
                           priorEmbeddedThings, additionalInfo)

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "AllEmbedders"),
                            ("embedders",
                            [x.getJsonableObject() for x in self.embedders])
                           ])


class RandomSubsetOfEmbedders(AbstractEmbedder):
    """Call some random subset of supplied embedders.

    Takes a quantity generator that generates a quantity of
    embedders, and executes that many embedders from a supplied set,
    in sequence

    Arguments:
        quantityGenerator: instance of :class:`.AbstractQuantityGenerator`

        embedders: a list of :class:`.AbstractEmbedder` objects
    """

    def __init__(self, quantityGenerator, embedders, name=None):
        if (isinstance(quantityGenerator, int)):
            quantityGenerator = FixedQuantityGenerator(quantityGenerator)
        assert isinstance(quantityGenerator, AbstractQuantityGenerator)
        self.quantityGenerator = quantityGenerator
        self.embedders = embedders
        super(RandomSubsetOfEmbedders, self).__init__(name)

    def _embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo):
        """See superclass.
        """
        numberOfEmbeddersToSample = self.quantityGenerator.generateQuantity()
        if (numberOfEmbeddersToSample > len(self.embedders)):
            raise RuntimeError("numberOfEmbeddersToSample came up as " + str(
                numberOfEmbeddersToSample) + " but total number of embedders is " + str(len(self.embedders)))
        sampledEmbedders = util.sampleWithoutReplacement(
            self.embedders, numberOfEmbeddersToSample)
        for embedder in sampledEmbedders:
            embedder.embed(backgroundStringArr,
                           priorEmbeddedThings, additionalInfo)

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "RandomSubsetOfEmbedders"), ("setOfEmbedders", [x.getJsonableObject() for x in self.embedders])])


class RepeatedEmbedder(AbstractEmbedder):
    """Call an embedded multiple times.

    Wrapper around an embedder to call it multiple times according to samples
    from a distribution. First calls ``self.quantityGenerator`` to get the
    quantity, then calls ``self.embedder`` a number of times equal
    to the value returned.

    Arguments:
        embedder: instance of :class:`.AbstractEmbedder`

        quantityGenerator: instance of :class:`.AbstractQuantityGenerator`
    """

    def __init__(self, embedder, quantityGenerator, name=None):
        self.embedder = embedder
        self.quantityGenerator = quantityGenerator
        super(RepeatedEmbedder, self).__init__(name)

    def _embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo):
        """See superclass.
        """
        quantity = self.quantityGenerator.generateQuantity()
        for i in range(quantity):
            self.embedder.embed(backgroundStringArr,
                                priorEmbeddedThings, additionalInfo)

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "RepeatedEmbedder"), ("embedder", self.embedder.getJsonableObject()), ("quantityGenerator", self.quantityGenerator.getJsonableObject())])


class AbstractQuantityGenerator(DefaultNameMixin):
    """Class for sampling values from a distribution.
    """

    def generateQuantity(self):
        """Sample a quantity from a distribution.

        Returns:
            The sampled value.
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


class ChooseValueFromASet(AbstractQuantityGenerator):
    """Randomly samples a particular value from a set of values.

    Arguments:
        setOfPossibleValues: array of values that will be randomly sampled
    from.

        name: see :class:`.DefaultNameMixin`.
    """

    def __init__(self, setOfPossibleValues, name=None):
        self.setOfPossibleValues = setOfPossibleValues
        super(ChooseValueFromASet, self).__init__(name)

    def generateQuantity(self):
        """See superclass.
        """
        return self.setOfPossibleValues[int(random.random() * (len(self.setOfPossibleValues)))]

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "ChooseValueFromASet"),
                            ("possibleValues", self.setOfPossibleValues)])


class UniformIntegerGenerator(AbstractQuantityGenerator):
    """Randomly samples an integer from minVal to maxVal, inclusive.

    Arguments:
        minVal: minimum integer that can be sampled

        maxVal: maximum integers that can be sampled
        
        name: See superclass.
    """

    def __init__(self, minVal, maxVal, name=None):
        self.minVal = minVal
        self.maxVal = maxVal
        super(UniformIntegerGenerator, self).__init__(name)

    def generateQuantity(self):
        """See superclass.
        """
        # the 1+ makes the max val inclusive
        return self.minVal + int(random.random() * (1 + self.maxVal - self.minVal))

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "UniformIntegerGenerator"), ("minVal", self.minVal), ("maxVal", self.maxVal)])


class FixedQuantityGenerator(AbstractQuantityGenerator):
    """Returns a fixed number every time generateQuantity is called.

    Arguments:
        quantity: the value to return when generateQuantity is called.
    """

    def __init__(self, quantity, name=None):
        self.quantity = quantity
        super(FixedQuantityGenerator, self).__init__(name)

    def generateQuantity(self):
        """See superclass.
        """
        return self.quantity

    def getJsonableObject(self):
        """See superclass.
        """
        return "fixedQuantity-" + str(self.quantity)


class PoissonQuantityGenerator(AbstractQuantityGenerator):
    """Generates values according to a poisson distribution.

    Arguments:
        mean: the mean of the poisson distribution
    """

    def __init__(self, mean, name=None):
        self.mean = mean
        super(PoissonQuantityGenerator, self).__init__(name)

    def generateQuantity(self):
        """See superclass.
        """
        return np.random.poisson(self.mean)

    def getJsonableObject(self):
        """See superclass.
        """
        return "poisson-" + str(self.mean)


class BernoulliQuantityGenerator(AbstractQuantityGenerator):
    """Generates 1 or 0 according to a bernoulli distribution.

    Arguments:
        prob: probability of 1
    """

    def __init__(self, prob, name=None):
        self.prob = prob
        super(BernoulliQuantityGenerator, self).__init__(name)

    def generateQuantity(self):
        """See superclass.
        """
        return 1 if (np.random.random() <= self.prob) else 0

    def getJsonableObject(self):
        """See sueprclass.
        """
        return "bernoulli-" + str(self.prob)


class MinMaxWrapper(AbstractQuantityGenerator):
    """Compress a distribution to lie within a min and a max.

    Wrapper that restricts a distribution to only return values between
    the min and the max. If a value outside the range is returned,
    resamples until it obtains a value within the range.
    Warns every time it tries to resample 10 times without successfully
    finding a value in the correct range.

    Arguments:
        quantityGenerator: instance of :class:`.AbstractQuantityGenerator`.
    Used to draw samples from the distribution to truncate

        theMin: can be None; if so will be ignored.

        theMax: can be None; if so will be ignored.
    """

    def __init__(self, quantityGenerator, theMin=None, theMax=None, name=None):
        self.quantityGenerator = quantityGenerator
        self.theMin = theMin
        self.theMax = theMax
        assert self.quantityGenerator is not None
        super(MinMaxWrapper, self).__init__(name)

    def generateQuantity(self):
        """See superclass.
        """
        tries = 0
        while (True):
            tries += 1
            quantity = self.quantityGenerator.generateQuantity()
            if ((self.theMin is None or quantity >= self.theMin) and (self.theMax is None or quantity <= self.theMax)):
                return quantity
            if (tries % 10 == 0):
                print("warning: made " + str(tries) +
                      " tries at trying to sample from distribution with min/max limits")

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("min", self.theMin), ("max", self.theMax), ("quantityGenerator", self.quantityGenerator.getJsonableObject())])


class ZeroInflater(AbstractQuantityGenerator):
    """Inflate a particular distribution with zeros.

    Wrapper that inflates the number of zeros returned.
    Flips a coin; if positive, will return zero - otherwise will
    sample from the wrapped distribution (which may still return 0)

    Arguments:
        quantityGenerator: an instance of :class:`.AbstractQuantityGenerator`;\
    represents the distribution to sample from with probability ``1-zeroProb``

        zeroProb: the probability of just returning 0 without sampling\
    from ``quantityGenerator``
        
        name: see :class:`.DefaultNameMixin`. 
    """

    def __init__(self, quantityGenerator, zeroProb, name=None):
        self.quantityGenerator = quantityGenerator
        self.zeroProb = zeroProb
        super(ZeroInflater, self).__init__(name)

    def generateQuantity(self):
        """See superclass.
        """
        val = random.random()
        if (val < self.zeroProb):
            return 0
        else:
            return self.quantityGenerator.generateQuantity()

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "ZeroInflater"), ("zeroProb", self.zeroProb), ("quantityGenerator", self.quantityGenerator.getJsonableObject())])


class SubstringEmbedder(EmbeddableEmbedder):
    """Used to embed substrings.

    Embeds a single generated substring within the background sequence,
    at a position sampled from a distribution. Only embeds at unoccupied
    positions

    Arguments:
        substringGenerator: instance of :class:`.AbstractSubstringGenerator`

        positionGenerator: instance of :class:`.AbstractPositionGenerator`

        name: see :class:`.DefaultNameMixin`.
    """

    def __init__(self, substringGenerator, positionGenerator=uniformPositionGenerator, name=None):
        super(SubstringEmbedder, self).__init__(
            SubstringEmbeddableGenerator(substringGenerator), positionGenerator, name)


def sampleIndexWithinRegionOfLength(length, lengthOfThingToEmbed):
    """Uniformly at random samples integers from 0 to
    ``length``-``lengthOfThingToEmbedIn``.

    Arguments:
        length: length of full region that could be embedded in

        lengthOfThingToEmbed: length of thing being embedded in larger region
    """
    assert lengthOfThingToEmbed <= length
    indexToSample = int(
        random.random() * ((length - lengthOfThingToEmbed) + 1))
    return indexToSample


class AbstractEmbeddableGenerator(DefaultNameMixin):
    """Generates an embeddable, usually for embedding in a background sequence.
    """

    def generateEmbeddable(self):
        """Generate an embeddable object.

        Returns:
            An instance of :class:`AbstractEmbeddable`
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


class PairEmbeddableGenerator(AbstractEmbeddableGenerator):
    """Embed a pair of embeddables with some separation.
        
    Arguments:
        emeddableGenerator1: instance of\
        :class:`.AbstractEmbeddableGenerator`. If an
        :class:`.AbstractSubstringGenerator` is provided, will be wrapped in\
        an instance of :class:`.SubstringEmbeddableGenerator`.

        embeddableGenerator2: same type information as for\
        ``embeddableGenerator1``

        separationGenerator: instance of\
        :class:`.AbstractQuantityGenerator`

        name: string, see :class:`DefaultNameMixin`
    """
    def __init__(self, embeddableGenerator1,
                 embeddableGenerator2, separationGenerator, name=None):
        if isinstance(embeddableGenerator1, AbstractSubstringGenerator):
            embeddableGenerator1 =\
                SubstringEmbeddableGenerator(embeddableGenerator1)
        if (isinstance(embeddableGenerator2, AbstractSubstringGenerator)):
            embeddableGenerator2 =\
                SubstringEmbeddableGenerator(embeddableGenerator2)
        self.embeddableGenerator1 = embeddableGenerator1
        self.embeddableGenerator2 = embeddableGenerator2
        self.separationGenerator = separationGenerator
        super(PairEmbeddableGenerator, self).__init__(name)

    def generateEmbeddable(self):
        """See superclass.
        """
        embeddable1 = self.embeddableGenerator1.generateEmbeddable()
        embeddable2 = self.embeddableGenerator2.generateEmbeddable()
        return PairEmbeddable(
            embeddable1=embeddable1, embeddable2=embeddable2,
            separation=self.separationGenerator.generateQuantity()
        )

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([
    ("class", "PairEmbeddableGenerator"),
    ("embeddableGenerator1", self.embeddableGenerator1.getJsonableObject()),
    ("embeddableenerator2", self.embeddableGenerator2.getJsonableObject()),
    ("separationGenerator", self.separationGenerator.getJsonableObject())])


class SubstringEmbeddableGenerator(AbstractEmbeddableGenerator):
    """Generates a :class:`.StringEmbeddable`

    Calls ``substringGenerator``, wraps the result in
    a :class:`.StringEmbeddable` and returns it.

    Arguments:
        substringGenerator: instance of :class:`.AbstractSubstringGenerator`
    """
    def __init__(self, substringGenerator, name=None):
        self.substringGenerator = substringGenerator
        super(SubstringEmbeddableGenerator, self).__init__(name)

    def generateEmbeddable(self):
        substring, substringDescription =\
            self.substringGenerator.generateSubstring()
        return StringEmbeddable(substring, substringDescription)

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "SubstringEmbeddableGenerator"),
    ("substringGenerator", self.substringGenerator.getJsonableObject())])


class AbstractSubstringGenerator(DefaultNameMixin):
    """
        Generates a substring, usually for embedding in a background sequence.
    """

    def generateSubstring(self):
        """
        Return:
            A tuple of ``(string, stringDescription)``; the result can be
        wrapped in an instance of :class:`.StringEmbeddable`.
        ``stringDescription`` is a short descriptor that does not contain
        spaces and may be prefixed in front of string when generating
        the __str__ representation for :class:`.StringEmbeddable`. 
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


class FixedSubstringGenerator(AbstractSubstringGenerator):
    """Generates the same string every time.

    When generateSubstring() is called, always returns the same string.
    The string also serves as its own description

    Arguments:
        fixedSubstring: the string to be generated

        name: see :class:`.DefaultNameMixin`
    """

    def __init__(self, fixedSubstring, name=None):
        self.fixedSubstring = fixedSubstring
        super(FixedSubstringGenerator, self).__init__(name)

    def generateSubstring(self):
        """See superclass.
        """
        return self.fixedSubstring, self.fixedSubstring

    def getJsonableObject(self):
        """See superclass.
        """
        return "fixedSubstring-" + self.fixedSubstring


class TransformedSubstringGenerator(AbstractSubstringGenerator):
    """Generates a substring and applies a series of transformations.

    Takes a substringGenerator and a set of AbstractTransformation objects,
    applies the transformations to the generated substring

    Arguments:
        substringGenerator: instance of :class:`.AbstractSubstringGenerator`

        transformations: an iterable of :class:`.AbstractTransformation`

        transformationsDescription: a string that will be prefixed in\
        front of ``substringDescription`` (generated by\
        ``substringGenerator.generateSubstring())`` to produce the\
        ``stringDescription``.

        name: see :class:`.DefaultNameMixin`.
    """

    def __init__(self, substringGenerator, transformations,
                       transformationsDescription="transformations",
                       name=None):
        self.substringGenerator = substringGenerator
        self.transformations = transformations
        self.transformationsDescription = transformationsDescription
        super(TransformedSubstringGenerator, self).__init__(self.name)

    def generateSubstring(self):
        """See superclass.
        """
        substring, substringDescription = self.substringGenerator.generateSubstring()
        baseSubstringArr = [x for x in substring]
        for transformation in self.transformations:
            baseSubstringArr = transformation.transform(baseSubstringArr)
        return "".join(baseSubstringArr), self.transformationsDescription + "-" + substringDescription

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "TransformedSubstringGenerator"),
    ("substringGenerator", self.substringGenerator.getJsonableObject()),
    ("transformations", [x.getJsonableObject() for x in self.transformations])
    ])


class AbstractTransformation(DefaultNameMixin):
    """Class representing a transformation applied to a character array.

    Takes an array of characters, applies some transformation.
    """

    def transform(self, stringArr):
        """Applies a transformation to stringArr.

            Arguments:
                stringArr: an array of characters.

            Returns:
                An array of characters that has the transformation applied.
            May mutate ``stringArr``
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


class RevertToReference(AbstractTransformation):
    """For a series of mutations, reverts the supplied character
    to the reference ("unmutated") string.

    Arguments:
        setOfMutations: instance of AbstractSetOfMutations

        name: see :class:`.DefaultNameMixin`.
    """

    def __init__(self, setOfMutations, name=None):
        self.setOfMutations = setOfMutations
        super(RevertToReference, self).__init__(name)

    def transform(self, stringArr):
        """See superclass.
        """
        for mutation in self.setOfMutations.getMutationsArr():
            mutation.revert(stringArr)
        return stringArr

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([
                 ("class", "RevertToReference"),
                 ("setOfMutations", self.setOfMutations.getJsonableObject())])


class AbstractApplySingleMutationFromSet(AbstractTransformation):
    """
        Class for applying a single mutation from a set of mutations; used
        to transform substrings generated by another method

        Arguments:
            setOfMutations: instance of :class:`.AbstractSetOfMutations`

            name: see :class:`.DefaultNameMixin`.
    """

    def __init__(self, setOfMutations, name=None):
        self.setOfMutations = setOfMutations
        super(AbstractApplySingleMutationFromSet, self).__init__(name)

    def transform(self, stringArr):
        """See superclass.
        """
        selectedMutation = self.selectMutation()
        selectedMutation.applyMutation(stringArr)
        return stringArr

    def selectMutation(self):
        """Chooses a mutation from the set of mutations to apply.

        Returns:
            an instance of :class:`.Mutation`
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([
    ("class", type(self).__name__),
    ("selectedMutations", self.setOfMutations.getJsonableObject())])


class Mutation(object):
    """Represent a single bp mutation in a motif sequence.

    Useful for creating simulations involving SNPs.

    Arguments:
        index: the position idx within the motif of the mutation

        previous: character, the previous base at this position

        new: character, the new base at this position after the mutation

        parentLength: optional; length of the motif. Used for assertion checks.
    """

    def __init__(self, index, previous, new, parentLength=None):
        self.index = index
        assert previous != new
        self.previous = previous
        self.new = new
        # the length of the full sequence that self.index indexes into
        self.parentLength = parentLength

    def parentLengthAssertionCheck(self, stringArr):
        """Checks that stringArr is consistent with parentLength if defined.
        """
        assert self.parentLength is None or len(stringArr) == self.parentLength

    def revert(self, stringArr):
        """Set the base at the position of the mutation to the unmutated value.

        Modifies stringArr which is an array of characters.

        Arguments:
            stringArr: an array of characters, which gets modified.
        """
        self.parentLengthAssertionCheck(stringArr)
        stringArr[self.index] = self.previous

    def applyMutation(self, stringArr):
        """Set the base at the position of the mutation to the mutated value.
        
        Modifies stringArr which is an array of characters.

        Arguments:
            stringArr: an array of characters, which gets modified.
        """
        self.parentLengthAssertionCheck(stringArr)
        assert stringArr[self.index] == self.previous
        stringArr[self.index] = self.new

class ChooseMutationAtRandom(AbstractApplySingleMutationFromSet):
    """Selects a mutation at random from self.setOfMutations to apply.
    """

    def selectMutation(self):
        mutationsArr = self.setOfMutations.getMutationsArr()
        return mutationsArr[int(random.random() * len(mutationsArr))]


class AbstractSetOfMutations(object):
    """Represents a collection of :class:`.Mutation` objects.

    Arguments:
        mutationsArr: array of :class:`.Mutation` objects
    """

    def __init__(self, mutationsArr):
        self.mutationsArr = mutationsArr

    def getMutationsArr(self):
        """Returns ``self.mutationsArr``

        Returns:
            ``self.mutationsArr``
        """
        return self.mutationsArr

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


class ReverseComplementWrapper(AbstractSubstringGenerator):
    """Reverse complements a string with a specified probability.

    Wrapper around an instance of
    :class:`.AbstractSubstringGenerator` that reverse complements the
    generated string with a specified probability.

    Arguments:
        substringGenerator: instance of `.AbstractSubstringGenerator`

        reverseComplementProb: probability of reverse complementation.
    Defaults to 0.5.

        name: see :class:`.DefaultNameMixin`.
    """

    def __init__(self, substringGenerator, reverseComplementProb=0.5, name=None):
        self.reverseComplementProb = reverseComplementProb
        self.substringGenerator = substringGenerator
        super(ReverseComplementWrapper, self).__init__(name)

    def generateSubstring(self):
        seq, seqDescription = self.substringGenerator.generateSubstring()
        if (random.random() < self.reverseComplementProb):
            seq = util.reverseComplement(seq)
            seqDescription = "revComp-" + seqDescription
        return seq, seqDescription

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([
    ("class", "ReverseComplementWrapper"),
    ("reverseComplementProb", self.reverseComplementProb),
    ("substringGenerator", self.substringGenerator.getJsonableObject())])


class PwmSampler(AbstractSubstringGenerator):
    """Samples from a pwm by calling ``self.pwm.sampleFromPwm``

    Arguments:
        pwm: an instance of ``pwm.PWM``

        name: see :class:`.DefaultNameMixin`
    """

    def __init__(self, pwm, name=None):
        self.pwm = pwm
        super(PwmSampler, self).__init__(name)

    def generateSubstring(self):
        """See superclass.
        """
        return self.pwm.sampleFromPwm()[0], self.pwm.name

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "PwmSampler"), ("motifName", self.pwm.name)])


class PwmSamplerFromLoadedMotifs(PwmSampler):
    """Instantiates a :class:`.PwmSampler` from a
    :class:`.LoadedEncodeMotifs` file.

    Convenience wrapper class for instantiating :class:`.PwmSampler`
    by pulling the pwm.PWM object using the provided name
    from an :class:`.AbstractLoadedMotifs` object

    Arguments:
        loadedMotifs: instance of :class:`.AbstractLoadedMotifs` 

        motifName: string, name of a motif in :class:`.AbstractLoadedMotifs`

        name: see :class:`.DefaultNameMixin`
    """

    def __init__(self, loadedMotifs, motifName, name=None):
        self.loadedMotifs = loadedMotifs
        super(PwmSamplerFromLoadedMotifs, self).__init__(
            loadedMotifs.getPwm(motifName), name)

    def getJsonableObject(self):
        """See superclass.
        """
        obj = super(PwmSamplerFromLoadedMotifs, self).getJsonableObject()
        return obj


class BestHitPwm(AbstractSubstringGenerator):
    """Always return the best possible match to a ``pwm.PWM`` when called.

    Arguments:
        pwm: an instance of ``pwm.PWM``

        bestHitMode: one of the values in ``pwm.BEST_HIT_MODE``. If ``pwmProb``
    then the best match will be determined according what is most likely
    to be sampled from the pwm matrix (this is the default). If
    ``logOdds``, then the best match will be determined according to what
    would result in the best match according to the log-odds matrix
    (so, taking the background into account).

        name: see :class:`.DefaultNameMixin`
    """

    def __init__(self, pwm, bestHitMode=pwm.BEST_HIT_MODE.pwmProb, name=None):
        self.pwm = pwm
        self.bestHitMode = bestHitMode
        super(BestHitPwm, self).__init__(name)

    def generateSubstring(self):
        """See superclass.
        """
        return self.pwm.getBestHit(self.bestHitMode), self.pwm.name

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "BestHitPwm"), ("pwm", self.pwm.name), ("bestHitMode", self.bestHitMode)])


class BestHitPwmFromLoadedMotifs(BestHitPwm):
    """Instantiates :class:`BestHitPwm` using a :class:`.LoadedMotifs` file.
    Analogous to :class:`.PwmSamplerFromLoadedMotifs`.
    """

    def __init__(self, loadedMotifs, motifName,
                 bestHitMode=pwm.BEST_HIT_MODE.pwmProb, name=None):
        self.loadedMotifs = loadedMotifs
        super(BestHitPwmFromLoadedMotifs, self).__init__(
            loadedMotifs.getPwm(motifName), bestHitMode, name)

    def getJsonableObject(self):
        """See superclass.
        """
        obj = super(BestHitPwmFromLoadedMotifs, self).getJsonableObject()
        return obj


class AbstractLoadedMotifs(object):
    """Class representing loaded PWMs.

    A class that contains instances of ``pwm.PWM`` loaded from a file.
    The pwms can be accessed by name.

    Arguments:
        loadedMotifs: dictionary mapping names of motifs
    to instances of ``pwm.PWM`` 
    """

    def __init__(self, loadedMotifs):
        self.loadedMotifs = loadedMotifs

    def getPwm(self, name):
        """Get a specific PWM.

        Returns:
            The ``pwm.PWM`` instance with the specified name.
        """
        return self.loadedMotifs[name]

    def addMotifs(self, abstractLoadedMotifs):
        """Adds the motifs in abstractLoadedMotifs to this.

        Arguments:
            abstractLoadedMotifs: instance of :class:`.AbstractLoadedMotifs`

        Returns:
            self, as a convenience
        """
        self.loadedMotifs.update(abstractLoadedMotifs.loadedMotifs)
        return self #convenience return


class AbstractLoadedMotifsFromFile(AbstractLoadedMotifs):
    """Class representing loaded PWMs.

    A class that contains instances of ``pwm.PWM`` loaded from a file.
    The pwms can be accessed by name.

    Arguments:
        fileName: string, the path to the file to load

        pseudocountProb: if some of the pwms have 0 probability for\
    some of the positions, will add the specified ``pseudocountProb``\
    to the rows of the pwm and renormalise.

        background: a dictionary with ACGT as the keys and the frequency as\
    the values. Defaults to ``util.DEFAULT_BACKGROUND_FREQ``
    """

    def __init__(self, fileName,
                       pseudocountProb=0.0,
                       background=util.DEFAULT_BACKGROUND_FREQ):
        self.fileName = fileName
        fileHandle = fp.getFileHandle(fileName)
        self.pseudocountProb = pseudocountProb
        self.background = background
        self.loadedMotifs = OrderedDict()
        action = self.getReadPwmAction(self.loadedMotifs)
        fp.performActionOnEachLineOfFile(
            fileHandle=fileHandle, transformation=fp.trimNewline, action=action
        )
        for pwm in self.loadedMotifs.values():
            pwm.finalise(pseudocountProb=self.pseudocountProb)
        super(AbstractLoadedMotifsFromFile, self).__init__(self.loadedMotifs)

    def getReadPwmAction(self, loadedMotifs):
        """Action performed when each line of the pwm text file is read in.

        This function is to be overridden by a specific implementation.
        It is executed on each line of the file when it is read in, and
        when PWMs are ready they will get inserted into ``loadedMotifs``.

        Arguments:
            loadedMotifs: an ``OrderedDict`` that will be filled with PWMs.
        The keys will be the names of the PWMs and the
        values will be instances of ``pwm.PWM``
        """
        raise NotImplementedError()


class LoadedEncodeMotifs(AbstractLoadedMotifsFromFile):
    """A class for reading in a motifs file in the ENCODE motifs format.

    This class is specifically for reading files in the encode motif
    format - specifically the motifs.txt file that contains Pouya's motifs
    (http://compbio.mit.edu/encode-motifs/motifs.txt)

    Basically, the motif declarations start with a >, the first
    characters after > until the first space are taken as the motif name,
    the lines after the line with a > have the format:
    "<ignored character> <prob of A> <prob of C> <prob of G> <prob of T>"
    """

    def getReadPwmAction(self, loadedMotifs):
        """See superclass.
        """
        currentPwm = util.VariableWrapper(None)

        def action(inp, lineNumber):
            if (inp.startswith(">")):
                inp = inp.lstrip(">")
                inpArr = inp.split()
                motifName = inpArr[0]
                currentPwm.var = pwm.PWM(motifName, background=self.background)
                loadedMotifs[currentPwm.var.name] = currentPwm.var
            else:
                # assume that it's a line of the pwm
                assert currentPwm.var is not None
                inpArr = inp.split()
                summaryLetter = inpArr[0]
                currentPwm.var.addRow([float(x) for x in inpArr[1:]])
        return action


class LoadedHomerMotifs(AbstractLoadedMotifsFromFile):
    """A class for reading in a motifs file in the Homer motifs format.

    Eg: HOCOMOCOv10_HUMAN_mono_homer_format_0.001.motif in resources
    """

    def getReadPwmAction(self, loadedMotifs):
        """See superclass.
        """
        currentPwm = util.VariableWrapper(None)

        def action(inp, lineNumber):
            if (inp.startswith(">")):
                inp = inp.lstrip(">")
                inpArr = inp.split()
                motifName = inpArr[1]
                currentPwm.var = pwm.PWM(motifName, background=self.background)
                loadedMotifs[currentPwm.var.name] = currentPwm.var
            else:
                # assume that it's a line of the pwm
                assert currentPwm.var is not None
                inpArr = inp.split()
                currentPwm.var.addRow([float(x) for x in inpArr[0:]])
        return action


class AbstractShuffler(object):
    """Implements a method to shuffle a supplied sequence"""

    def shuffle(self, string):
        raise NotImplementedError()

    def getJsonableObject(self):
        return OrderedDict([('class', type(self).__name__)])


class DinucleotideShuffler(AbstractShuffler):

    def shuffle(self, string):
        return dinuc_shuffle.dinuc_shuffle(string)  


class AbstractBackgroundGenerator(object):
    """Returns the sequence that :class:`.AbstractEmbeddable` objects
    are to be embedded into.
    """

    def generateBackground(self):
        """Returns a sequence that is the background.
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        raise NotImplementedError()


class ShuffledBackgroundGenerator(AbstractBackgroundGenerator):
    """Shuffles a given sequence

    Arguments:
        string: the string to shuffle 
        shuffler: instance of :class:`.AbstractShuffler`.

    Returns:
        The shuffled string
    """
    def __init__(self, string, shuffler):
        self.string = string
        self.shuffler = shuffler

    def generateBackground(self):
        return self.shuffler.shuffle(self.string)

    def getJsonableObject(self):
        """See superclass.
        """
        raise NotImplementedError()


class RepeatedSubstringBackgroundGenerator(AbstractBackgroundGenerator):
    """Repeatedly call a substring generator and concatenate the result.

    Can be used to generate variable-length sequences.

    Arguments:
        substringGenerator: instance of :class:`.AbstractSubstringGenerator`

        repetitions: instance of :class:`.AbstractQuantityGenerator`.\
        If pass an int, will create a\
        :class:`.FixedQuantityGenerator` from the int. This will be called\
        to determine the number of times to generate a substring from\
        ``self.substringGenerator``

    Returns:
        The concatenation of all the calls to ``self.substringGenerator``
    """

    def __init__(self, substringGenerator, repetitions):
        self.substringGenerator = substringGenerator
        if isinstance(repetitions, int):
            self.repetitions = FixedQuantityGenerator(repetitions)
        else:
            assert isinstance(repetitions, AbstractQuantityGenerator)
            self.repetitions = repetitions

    def generateBackground(self):
        toReturn = []
        for i in range(self.repetitions.generateQuantity()):
            # first pos is substring, second pos is the name
            toReturn.append(self.substringGenerator.generateSubstring()[0])
        return "".join(toReturn)

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([
    ("class", "RepeatedSubstringBackgroundGenerator"),
    ("substringGenerator", self.substringGenerator.getJsonableObject()),
    ("repetitions", self.repetitions.getJsonableObject())])


class SampleFromDiscreteDistributionSubstringGenerator(AbstractSubstringGenerator):
    """Generate a substring by sampling from a distribution.

    If the "substrings" are single characters (A/C/G/T), can be used
    in conjunction with :class:`.RepeatedSubstringBackgroundGenerator` to
    generate sequences with a certain GC content.

    Arguments:
        discreteDistribution: instance of ``util.DiscreteDistribution``
    """

    def __init__(self, discreteDistribution):
        self.discreteDistribution = discreteDistribution

    def generateSubstring(self):
        return self.discreteDistribution.sample()

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([
    ("class", "SampleFromDiscreteDistributionSubstringGenerator"),
    ("discreteDistribution", self.discreteDistribution.valToFreq)])


class ZeroOrderBackgroundGenerator(RepeatedSubstringBackgroundGenerator):
    """Returns a sequence with a certain GC content.

    Each base is sampled independently.

    Arguments:
        seqLength: int, length of the background 

        discreteDistribution: either an instance of\
    ``util.DiscreteDistribution` or a dict mapping values to frequency.\
    defaults to ``util.DEFAULT_BASE_DISCRETE_DISTRIBUTION``
    """

    def __init__(self, seqLength,
        discreteDistribution=util.DEFAULT_BASE_DISCRETE_DISTRIBUTION):
        if isinstance(discreteDistribution,dict):
            discreteDistribution=util.DiscreteDistribution(
                discreteDistribution)
        super(ZeroOrderBackgroundGenerator, self).__init__(
    SampleFromDiscreteDistributionSubstringGenerator(discreteDistribution),
    seqLength)


class FirstOrderBackgroundGenerator(AbstractBackgroundGenerator):
    """Returns a sequence from a first order markov chain with defined
    gc content

    Each base is sampled independently.

    Arguments:
        seqLength: int, length of the background 
        priorFrequencies: ordered dictionary with freqs of starting base
        dinucFrequencies: dictionary with the frequences of the dinucs
    """

    def __init__(self,
                 seqLength,
                 priorFrequencies=util.DEFAULT_BACKGROUND_FREQ,
                 dinucFrequencies=util.DEFAULT_DINUC_FREQ):
        self.seqLength = seqLength
        assert self.seqLength > 0

        #do some sanity checks on dinucFrequencies
        assert abs(sum(dinucFrequencies.values())-1.0) < 10**-7,\
         sum(dinucFrequencies.values())
        assert all(len(key)==2 for key in dinucFrequencies.keys())

        #build a transition matrix and priors matrix
        chars = set([key[0] for key in dinucFrequencies]) 
        transitionMatrix = {}
        for char in chars:
            probOnSecondChar = OrderedDict()
            totalProb = 0.0
            for key in dinucFrequencies:
                if key[0]==char:
                    probOnSecondChar[key[1]] = dinucFrequencies[key]
                    totalProb += probOnSecondChar[key[1]]
            probOnSecondChar = util.DiscreteDistribution(
                OrderedDict([(key,val/totalProb) for key,val
                in probOnSecondChar.items()]))
            transitionMatrix[char] = probOnSecondChar

        self.transitionMatrix = transitionMatrix
        self.priorFrequencies = util.DiscreteDistribution(priorFrequencies)

    def generateBackground(self):
        generatedCharacters = [self.priorFrequencies.sample()]
        for i in range(self.seqLength-1):
            generatedCharacters.append(
                self.transitionMatrix[generatedCharacters[-1]].sample())
        return "".join(generatedCharacters)
