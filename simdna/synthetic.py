from __future__ import absolute_import, division, print_function
from . import util, pwm, fileProcessing as fp
import argparse
import numpy as np
import random
import math
from collections import OrderedDict
import json
import re


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


class DefaultNameMixin(object):
    """Basic functionality for classes that have a self.name attribute
    
    Arguments:
        name: string
    """
    def __init__(self, name):
        if (name == None):
            name = self.getDefaultName()
        self.name = name

    def getDefaultName(self):
        return RuntimeError("No default name implementation")


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
        what: object representing the thing that has been embedded.
            Should have __str__ and __len__ defined. Often is an instance
            of :class:`.AbstractEmbeddable`
        startPos: int, the position relative to the start of the parent
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
            string: assumed to have format:
                description[-|_]startPos[-|_]whatString, where
                whatString will be provided to whatClass
            whatClass: the class (usually a :class:`.AbstractEmbeddable`) that
                will be used to instantiate the what from the whatString

        Returns:
            The Embedding class called with
            what=whatClass.fromString(whatString), startPos=int(startPos)
        """
        if (whatClass is None):
            whatClass = StringEmbeddable
        # was printed out as pos-[startPos]_[what], but the
        #[what] may contain underscores, hence the maxsplit
        # to avoid splitting on them.
        prefix, startPos, whatString = re.split("-|_", string, maxsplit=2)
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
        self.sequenceCounter = 0

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

    def generateSequence(self):
        """Produce the sequence.
        
        Generates a background using self.backgroundGenerator,
        splits it into an array, and passes it to each of
        self.embedders in turn for embedding things.

        Returns:
            An instance of :class:`.GeneratedSequence`
        """
        additionalInfo = AdditionalInfo()
        backgroundString = self.backgroundGenerator.generateBackground()
        backgroundStringArr = [x for x in backgroundString]
        # priorEmbeddedThings keeps track of what has already been embedded
        priorEmbeddedThings = PriorEmbeddedThings_numpyArrayBacked(
            len(backgroundStringArr))
        for embedder in self.embedders:
            embedder.embed(backgroundStringArr,
                           priorEmbeddedThings, additionalInfo)
        self.sequenceCounter += 1
        return GeneratedSequence(self.namePrefix + str(self.sequenceCounter),
                                 "".join(backgroundStringArr),
                                 priorEmbeddedThings.getEmbeddings(),
                                 additionalInfo)

    def getJsonableObject(self):
        """See parent.
        """
        return OrderedDict([("class", "EmbedInABackground"),
                            ("namePrefix", self.namePrefix),
                            ("backgroundGenerator",
                             self.backgroundGenerator.getJsonableObject()),
                            ("embedders",
                             [x.getJsonableObject() for x in self.embedders])
                            ])


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
    See parent for more documentation.

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

        stringDescription: a short descriptor prefixed before the
        __str__ representation of the embeddable. Should not contain a hyphen.
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

    def embedInBackgroundStringArr(self, priorEmbeddedThings, backgroundStringArr, startPos):
        """See superclass.
        """
        backgroundStringArr[startPos:startPos + len(self.string)] = self.string
        priorEmbeddedThings.addEmbedding(startPos, self)

    @classmethod
    def fromString(cls, theString):
        """Generates a StringEmbeddable from the provided string.

        Arguments:
            theString: string of the format ``stringDescription-coreString``.
        Will then return:
        ``StringEmbeddable(string=coreString,
                             stringDescription=stringDescription)``

        Returns:
            An instance of :class:`.StringEmbeddable`
        """
        if ("-" in theString):
            stringDescription, coreString = theString.split("-")
            return cls(string=coreString, stringDescription=stringDescription)
        else:
            return cls(string=theString)


class PairEmbeddable_General(AbstractEmbeddable):
    """Embed two embeddables with some separation.

    Arguments:
        embeddable1: first embeddable to be embedded

        embeddable2: second embeddable to be embedded

        separation: int of positions separating embeddable1 and embeddable2

        embeddableDescription: a concise descriptive string prefixed in
        front when generating a __str__ representation of the embeddable.
        Should not contain a hyphen.

        nothingInBetween: if true, then nothing else is allowed to be
        embedded in the gap between embeddable1 and embeddable2.
    """

    def __init__(self, embeddable1, embeddable2, separation,
                       embeddableDescription, nothingInBetween=True):
        self.embeddable1 = embeddable1
        self.embeddable2 = embeddable2
        self.separation = separation
        self.embeddableDescription = embeddableDescription
        self.nothingInBetween = nothingInBetween

    def __len__(self):
        return len(self.embeddable1) + self.separation + len(self.embeddable2)

    def __str__(self):
        return self.embeddableDescription + ("-" if embeddableDescription != "" else "") + str(self.embeddable1) + "-Gap" + str(self.separation) + "-" + str(self.embeddable2)

    def getDescription(self):
        return self.embeddableDescription

    def canEmbed(self, priorEmbeddedThings, startPos):
        if (self.nothingInBetween):
            return priorEmbeddedThings.canEmbed(startPos, startPos + len(self))
        else:
            return (priorEmbeddedThings.canEmbed(startPos, startPos + len(self.embeddable1))
                    and priorEmbeddedThings.canEmbed(startPos + len(self.embeddable1) + self.separation, startPos + len(self)))

    def embedInBackgroundStringArr(self, priorEmbeddedThings,
                                         backgroundStringArr, startPos):
        self.embeddable1.embedInBackgroundStringArr(
            priorEmbeddedThings, backgroundStringArr, startPos)
        self.embeddable2.embedInBackgroundStringArr(
            priorEmbeddedThings, backgroundStringArr, startPos + self.separation)
        if (self.nothingInBetween):
            priorEmbeddedThings.addEmbedding(startPos, self)
        else:
            priorEmbeddedThings.addEmbedding(startPos, self.embeddable1)
            priorEmbeddedThings.addEmbedding(
                startPos + len(self.string1) + self.separation, self.embeddable2)


class PairEmbeddable(AbstractEmbeddable):
    """Embed two strings with some separation. To be deprecated.

    To be deprecated in favour of PairEmbeddable_General.

    Arguments:
        string1: first string to be embedded

        string2: second string to be embedded

        separation: int of positions separating string1 and string2

        embeddableDescription: a concise descriptive string prefixed in
        front when generating a __str__ representation of the embeddable.
        Should not contain a hyphen.

        nothingInBetween: if true, then nothing else is allowed to be
        embedded in the gap between string1 and string2.
    """

    def __init__(self, string1, string2, separation, embeddableDescription, nothingInBetween=True):
        self.string1 = string1
        self.string2 = string2
        self.separation = separation
        self.embeddableDescription = embeddableDescription
        self.nothingInBetween = nothingInBetween

    def __len__(self):
        return len(self.string1) + self.separation + len(self.string2)

    def __str__(self):
        return self.embeddableDescription + "-" + self.string1 + "-Gap" + str(self.separation) + "-" + self.string2

    def getDescription(self):
        return self.embeddableDescription

    def canEmbed(self, priorEmbeddedThings, startPos):
        if (self.nothingInBetween):
            return priorEmbeddedThings.canEmbed(startPos, startPos + len(self))
        else:
            return (priorEmbeddedThings.canEmbed(startPos, startPos + len(self.string1))
                    and priorEmbeddedThings.canEmbed(startPos + len(self.string1) + self.separation, startPos + len(self)))

    def embedInBackgroundStringArr(self, priorEmbeddedThings, backgroundStringArr, startPos):
        backgroundStringArr[startPos:startPos +
                            len(self.string1)] = self.string1
        backgroundStringArr[
            startPos + len(self.string1) + self.separation:startPos + len(self)] = self.string2
        if (self.nothingInBetween):
            priorEmbeddedThings.addEmbedding(startPos, self)
        else:
            priorEmbeddedThings.addEmbedding(startPos, self.string1)
            priorEmbeddedThings.addEmbedding(
                startPos + len(self.string1) + self.separation, self.string2)


class AbstractEmbedder(DefaultNameMixin):
    """Produces :class:`AbstractEmbeddable` objects and
    embeds them in a sequence.
    """

    def embed(self, backgroundStringArr, priorEmbeddedThings, additionalInfo=None):
        """Embeds things in the provided ``backgroundStringArr``.

        Modifies backgroundStringArr to include whatever has been embedded.

        Arguments:
            backgroundStringArr: array of characters
        representing the background string

            priorEmbeddedThings: instance of
        :class:`.AbstractPriorEmbeddedThings`

            additionalInfo: instance of :class:`.AdditionalInfo`;
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
                print("Warning: made " + str(tries) + " at trying to embed " + str(embeddable) + " in region of length " + str(
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
        return OrderedDict([("class", "ChooseValueFromASet"), ("possibleValues", self.setOfPossibleValues)])


class UniformIntegerGenerator(AbstractQuantityGenerator):
    """Randomly samples an integer from minVal to maxVal, inclusive.
    """

    def __init__(self, minVal, maxVal, name=None):
        self.minVal = minVal
        self.maxVal = maxVal
        super(UniformIntegerGenerator, self).__init__(name)

    def generateQuantity(self):
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
        return 1 if (np.random.random() <= self.prob) else 0

    def getJsonableObject(self):
        """See sueprclass.
        """
        return "bernoulli-" + str(self.prob)


class MinMaxWrapper(AbstractQuantityGenerator):
    """Compress a distribution to lie within a min and a max.

    Wrapper that restricts a distribution to only return values between the min and
        the max. If a value outside the range is returned, resamples until
        it obtains a value within the range. Warns if it resamples too many times.

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
        quantityGenerator: an instance of :class:`.AbstractQuantityGenerator`;
            represents the distribution to sample from with probability
            1-zeroProb
        zeroProb: the probability of just returning 0
            without sampling from quantityGenerator
    """

    def __init__(self, quantityGenerator, zeroProb, name=None):
        self.quantityGenerator = quantityGenerator
        self.zeroProb = zeroProb
        super(ZeroInflater, self).__init__(name)

    def generateQuantity(self):
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
    """

    def __init__(self, substringGenerator, positionGenerator=uniformPositionGenerator, name=None):
        super(SubstringEmbedder, self).__init__(
            SubstringEmbeddableGenerator(substringGenerator), positionGenerator, name)


def sampleIndexWithinRegionOfLength(length, lengthOfThingToEmbed):
    """Uniformly at random samples integers from 0 to
    length-lengthOfThingToEmbedIn.
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


class PairEmbeddableGenerator_General(AbstractEmbeddableGenerator):
    """Embed a pair of embeddables with some separation. This class needs
        to eventually replace :class:`.PairEmbeddableGenerator`
        
    Arguments:
        emeddableGenerator1: instance of
            :class:`.AbstractEmbeddableGenerator`

        embeddableGenerator2: instance of
            :class:`.AbstractEmbeddableGenerator`

        separationGenerator: instance of
            :class:`.AbstractQuantityGenerator`

        name: string, see :class:`DefaultNameMixin`
    """
    def __init__(self, embeddableGenerator1, embeddableGenerator2, separationGenerator, name=None):
        self.embeddableGenerator1 = embeddableGenerator1
        self.embeddableGenerator2 = embeddableGenerator2
        self.separationGenerator = separationGenerator
        super(PairEmbeddableGenerator, self).__init__(name)

    def generateEmbeddable(self):
        """See superclass.
        """
        embeddable1 = self.embeddableGenerator1.generateEmbeddable()
        embeddable2 = self.embeddableGenerator2.generateEmbeddable()
        return PairEmbeddable_General(
            embeddable1, embeddable2,
             self.separationGenerator.generateQuantity(
            ), embeddable1.getDescription() + "+" + embeddable2.getDescription()
        )

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "PairEmbeddableGenerator"), ("embeddableGenerator1", self.embeddableGenerator1.getJsonableObject()), ("embeddableenerator2", self.embeddableGenerator2.getJsonableObject()), ("separationGenerator", self.separationGenerator.getJsonableObject())
                            ])


class PairEmbeddableGenerator(AbstractEmbeddableGenerator):
    """Embed a pair of substrings with some separation. This class needs
        to be deprecated in favour of just using
        :class:`.PairEmbeddableGenerator_General`
        
    Arguments:
        substringGenerator1: instance of
            :class:`.AbstractSubstringGenerator`

        substringGenerator2: instance of
            :class:`.AbstractSubstringGenerator`

        separationGenerator: instance of
            :class:`.AbstractQuantityGenerator`

        name: string, see :class:`DefaultNameMixin`
    """

    def __init__(self, substringGenerator1, substringGenerator2, separationGenerator, name=None):
        self.substringGenerator1 = substringGenerator1
        self.substringGenerator2 = substringGenerator2
        self.separationGenerator = separationGenerator
        super(PairEmbeddableGenerator, self).__init__(name)

    def generateEmbeddable(self):
        """See superclass.
        """
        string1, string1Description = self.substringGenerator1.generateSubstring()
        string2, string2Description = self.substringGenerator2.generateSubstring()
        return PairEmbeddable(
            string1, string2, self.separationGenerator.generateQuantity(), string1Description +
            "+" + string2Description
        )

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "PairEmbeddableGenerator"), ("substringGenerator1", self.substringGenerator1.getJsonableObject()), ("substringGenerator2", self.substringGenerator2.getJsonableObject()), ("separationGenerator", self.separationGenerator.getJsonableObject())
                            ])


class SubstringEmbeddableGenerator(AbstractEmbeddableGenerator):
    """
    Arguments:
        substringGenerator: instance of :class:`.AbstractSubstringGenerator`
    """
    def __init__(self, substringGenerator, name=None):
        self.substringGenerator = substringGenerator
        super(SubstringEmbeddableGenerator, self).__init__(name)

    def generateEmbeddable(self):
        substring, substringDescription = self.substringGenerator.generateSubstring()
        return StringEmbeddable(substring, substringDescription)

    def getJsonableObject(self):
        return OrderedDict([("class", "SubstringEmbeddableGenerator"), ("substringGenerator", self.substringGenerator.getJsonableObject())])


class AbstractSubstringGenerator(DefaultNameMixin):
    """
        Generates a substring, usually for embedding in a background sequence.
    """

    def generateSubstring(self):
        raise NotImplementedError()

    def getJsonableObject(self):
        raise NotImplementedError()


class FixedSubstringGenerator(AbstractSubstringGenerator):
    """
        When generateSubstring() is called, always returns the same string.
        The string also serves as its own description
    """

    def __init__(self, fixedSubstring, name=None):
        self.fixedSubstring = fixedSubstring
        super(FixedSubstringGenerator, self).__init__(name)

    def generateSubstring(self):
        return self.fixedSubstring, self.fixedSubstring

    def getJsonableObject(self):
        return "fixedSubstring-" + self.fixedSubstring


class TransformedSubstringGenerator(AbstractSubstringGenerator):
    """
        Takes a substringGenerator and a set of AbstractTransformation objects,
        applies the transformations to the generated substring
    """

    def __init__(self, substringGenerator, transformations, transformationsDescription="transformations", name=None):
        self.substringGenerator = substringGenerator
        self.transformations = transformations
        self.transformationsDescription = transformationsDescription
        super(TransformedSubstringGenerator, self).__init__(self.name)

    def generateSubstring(self):
        substring, substringDescription = self.substringGenerator.generateSubstring()
        baseSubstringArr = [x for x in substring]
        for transformation in self.transformations:
            baseSubstringArr = transformation.transform(baseSubstringArr)
        return "".join(baseSubstringArr), self.transformationsDescription + "-" + substringDescription

    def getJsonableObject(self):
        return OrderedDict([("class", "TransformedSubstringGenerator"), ("substringGenerator", self.substringGenerator.getJsonableObject()), ("transformations", [x.getJsonableObject() for x in self.transformations])])


class AbstractTransformation(DefaultNameMixin):
    """
        takes an array of characters, applies some transformation, returns an
        array of characters (may be the same (mutated) one or a different one)
    """

    def transform(self, stringArr):
        """
            stringArr is an array of characters.
            Returns an array of characters that has the transformation applied.
            May mutate stringArr
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        raise NotImplementedError()


class RevertToReference(AbstractTransformation):
    """
        for a series of mutations, reverts the supplied string to the reference
        ("unmutated") string
    """

    def __init__(self, setOfMutations, name=None):
        """
            setOfMutations: instance of AbstractSetOfMutations
        """
        self.setOfMutations = setOfMutations
        super(RevertToReference, self).__init__(name)

    def transform(self, stringArr):  # see parent docs
        for mutation in self.setOfMutations.getMutationsArr():
            mutation.revert(stringArr)
        return stringArr

    def getJsonableObject(self):
        return OrderedDict([("class", "RevertToReference"), ("setOfMutations", self.setOfMutations.getJsonableObject())])


class AbstractApplySingleMutationFromSet(AbstractTransformation):
    """
        Class for applying a single mutation from a set of mutations; used
        to transform substrings generated by another method
    """

    def __init__(self, setOfMutations, name=None):
        """
            setOfMutations: instance of AbstractSetOfMutations
        """
        self.setOfMutations = setOfMutations
        super(AbstractApplySingleMutationFromSet, self).__init__(name)

    def transform(self, stringArr):  # see parent docs
        selectedMutation = self.selectMutation()
        selectedMutation.applyMutation(stringArr)
        return stringArr

    def selectMutation(self):
        raise NotImplementedError()

    def getClassName(self):
        raise NotImplementedError()

    def getJsonableObject(self):
        return OrderedDict([("class", self.getClassName()), ("selectedMutations", self.setOfMutations.getJsonableObject())])


class ChooseMutationAtRandom(AbstractApplySingleMutationFromSet):
    """
        Selects a mutation at random from self.setOfMutations to apply; see parent docs.
    """

    def selectMutation(self):
        mutationsArr = self.setOfMutations.getMutationsArr()
        return mutationsArr[int(random.random() * len(mutationsArr))]

    def getClassName(self):
        return "ChooseMutationAtRandom"


class AbstractSetOfMutations(object):
    """
        Represents a collection of pwm.Mutation objects
    """

    def __init__(self, mutationsArr):
        """
            mutationsArr: array of pwm.Mutation objects
        """
        self.mutationsArr = mutationsArr

    def getMutationsArr(self):
        return self.mutationsArr

    def getJsonableObject(self):
        raise NotImplementedError()


class TopNMutationsFromPwmRelativeToBestHit(AbstractSetOfMutations):
    """
        See docs for parent; here, the collection of mutations are the
        top N strongest mutations for a PWM as compared to the best
        match for that pwm.
    """

    def __init__(self, pwm, N, bestHitMode):
        """
            pwm: instance of pwm.PWM
            N: the N in the top N strongest mutations
            bestHitMode: one of pwm.BEST_HIT_MODE; pwm.BEST_HIT_MODE.pwmProb defines the
                topN mutations relative to the probability matrix of the pwm, while
                pwm.BEST_HIT_MODE.logOdds defines the topN mutations relative to the log
                odds matrix computed using the background frequency specified in the
                pwm object.
        """
        self.pwm = pwm
        self.N = N
        self.bestHitMode = bestHitMode
        mutationsArr = self.pwm.computeSingleBpMutationEffects(
            self.bestHitMode)
        super(TopNMutationsFromPwmRelativeToBestHit,
              self).__init__(mutationsArr)

    def getJsonableObject(self):
        return OrderedDict([("class", "TopNMutationsFromPwmRelativeToBestHit"), ("pwm", self.pwm.name), ('N', self.N), ("bestHitMode", self.bestHitMode)])

# is an AbstractSetOfMutations object


class TopNMutationsFromPwmRelativeToBestHit_FromLoadedMotifs(TopNMutationsFromPwmRelativeToBestHit):
    """
        Like parent, except extracts the pwm.PWM object from an AbstractLoadedMotifs object,
        saving you a few lines of code.
    """

    def __init__(self, loadedMotifs, pwmName, N, bestHitMode):
        self.loadedMotifs = loadedMotifs
        super(TopNMutationsFromPwmRelativeToBestHit_FromLoadedMotifs, self).__init__(
            self.loadedMotifs.getPwm(pwmName), N, bestHitMode)

    def getJsonableObject(self):
        obj = super(
            TopNMutationsFromPwmRelativeToBestHit_FromLoadedMotifs, self).getJsonableObject()
        obj['loadedMotifs'] = self.loadedMotifs.getJsonableObject()
        return obj


class ReverseComplementWrapper(AbstractSubstringGenerator):
    """
        Wrapper around a AbstractSubstringGenerator that reverse complements it
        with the specified probability.
    """

    def __init__(self, substringGenerator, reverseComplementProb=0.5, name=None):
        """
            substringGenerator: instance of AbstractSubstringGenerator
            reverseComplementProb: probability of reverse complementing it.
        """
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
        return OrderedDict([("class", "ReverseComplementWrapper"), ("reverseComplementProb", self.reverseComplementProb), ("substringGenerator", self.substringGenerator.getJsonableObject())])


class PwmSampler(AbstractSubstringGenerator):
    """
        samples from the pwm by calling self.pwm.sampleFromPwm
    """

    def __init__(self, pwm, name=None):
        self.pwm = pwm
        super(PwmSampler, self).__init__(name)

    def generateSubstring(self):
        return self.pwm.sampleFromPwm()[0], self.pwm.name

    def getJsonableObject(self):
        return OrderedDict([("class", "PwmSampler"), ("motifName", self.pwm.name)])


class PwmSamplerFromLoadedMotifs(PwmSampler):
    """
        convenience wrapper class for instantiating parent by pulling the pwm given the name
        from an AbstractLoadedMotifs object (it basically extracts the pwm for you)
    """

    def __init__(self, loadedMotifs, motifName, name=None):
        self.loadedMotifs = loadedMotifs
        super(PwmSamplerFromLoadedMotifs, self).__init__(
            loadedMotifs.getPwm(motifName), name)

    def getJsonableObject(self):
        obj = super(PwmSamplerFromLoadedMotifs, self).getJsonableObject()
        obj['loadedMotifs'] = self.loadedMotifs.getJsonableObject()
        return obj


class BestHitPwm(AbstractSubstringGenerator):
    """
        always returns the best possible match to the pwm in question when called
    """

    def __init__(self, pwm, bestHitMode=pwm.BEST_HIT_MODE.pwmProb, name=None):
        self.pwm = pwm
        self.bestHitMode = bestHitMode
        super(BestHitPwm, self).__init__(name)

    def generateSubstring(self):
        return self.pwm.getBestHit(self.bestHitMode), self.pwm.name

    def getJsonableObject(self):
        return OrderedDict([("class", "BestHitPwm"), ("pwm", self.pwm.name), ("bestHitMode", self.bestHitMode)])


class BestHitPwmFromLoadedMotifs(BestHitPwm):
    """
        convenience wrapper class for instantiating parent by pulling the pwm given the name
        from an AbstractLoadedMotifs object (it basically extracts the pwm for you)
    """

    def __init__(self, loadedMotifs, motifName, bestHitMode=pwm.BEST_HIT_MODE.pwmProb, name=None):
        self.loadedMotifs = loadedMotifs
        super(BestHitPwmFromLoadedMotifs, self).__init__(
            loadedMotifs.getPwm(motifName), bestHitMode, name)

    def getJsonableObject(self):
        obj = super(BestHitPwmFromLoadedMotifs, self).getJsonableObject()
        obj['loadedMotifs'] = self.loadedMotifs.getJsonableObject()
        return obj


class AbstractLoadedMotifs(object):
    """
        A class that contains instances of pwm.PWM loaded from a file.
        The pwms can be accessed by name.
    """

    def __init__(self, fileName, pseudocountProb=0.0, background=util.DEFAULT_BACKGROUND_FREQ):
        """
            fileName: the path to the file to laod
            pseudocountProb: if some of the pwms have 0 probability for
            some of the positions, will add the specified pseudocountProb
            to the rows of the pwm and renormalise.
        """
        self.fileName = fileName
        fileHandle = fp.getFileHandle(fileName)
        self.pseudocountProb = pseudocountProb
        self.background = background
        self.recordedPwms = OrderedDict()
        action = self.getReadPwmAction(self.recordedPwms)
        fp.performActionOnEachLineOfFile(
            fileHandle=fileHandle, transformation=fp.trimNewline, action=action
        )
        for pwm in self.recordedPwms.values():
            pwm.finalise(pseudocountProb=self.pseudocountProb)

    def getPwm(self, name):
        """
            returns the pwm.PWM instance with the specified name.
        """
        return self.recordedPwms[name]

    def getReadPwmAction(self, recordedPwms):
        """
            This is the action that is to be performed on each line of the
            file when it is read in. recordedPwms is an OrderedDict that
            stores instances of pwm.PWM
        """
        raise NotImplementedError()

    def getJsonableObject(self):
        return OrderedDict([("fileName", self.fileName), ("pseudocountProb", self.pseudocountProb), ("background", self.background)])


class LoadedEncodeMotifs(AbstractLoadedMotifs):
    """
        This class is specifically for reading files in the encode motif
        format - specifically the motifs.txt file that contains Pouya's motifs
    """

    def getReadPwmAction(self, recordedPwms):
        currentPwm = util.VariableWrapper(None)

        def action(inp, lineNumber):
            if (inp.startswith(">")):
                inp = inp.lstrip(">")
                inpArr = inp.split()
                motifName = inpArr[0]
                currentPwm.var = pwm.PWM(motifName, background=self.background)
                recordedPwms[currentPwm.var.name] = currentPwm.var
            else:
                # assume that it's a line of the pwm
                assert currentPwm.var is not None
                inpArr = inp.split()
                summaryLetter = inpArr[0]
                currentPwm.var.addRow([float(x) for x in inpArr[1:]])
        return action


class AbstractBackgroundGenerator(object):
    """
        Returns the sequence that the embeddings are subsequently inserted into.
    """

    def generateBackground(self):
        raise NotImplementedError()

    def getJsonableObject(self):
        raise NotImplementedError()


class RepeatedSubstringBackgroundGenerator(AbstractBackgroundGenerator):

    def __init__(self, substringGenerator, repetitions):
        """
            substringGenerator: instance of AbstractSubstringGenerator
            repetitions: instance of AbstractQuantityGenerator. If pass an int,
                will create a FixedQuantityGenerator from the int.
            returns the concatenation of all the calls to the substringGenerator
        """
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
        return OrderedDict([("class", "RepeatedSubstringBackgroundGenerator"), ("substringGenerator", self.substringGenerator.getJsonableObject()), ("repetitions", self.repetitions.getJsonableObject())])


class SampleFromDiscreteDistributionSubstringGenerator(AbstractSubstringGenerator):

    def __init__(self, discreteDistribution):
        """
            discreteDistribution: instance of util.DiscreteDistribution
        """
        self.discreteDistribution = discreteDistribution

    def generateSubstring(self):
        return util.sampleFromDiscreteDistribution(self.discreteDistribution)

    def getJsonableObject(self):
        return OrderedDict([("class", "SampleFromDiscreteDistributionSubstringGenerator"), ("discreteDistribution", self.discreteDistribution.valToFreq)])


class ZeroOrderBackgroundGenerator(RepeatedSubstringBackgroundGenerator):
    """
        returns a sequence with 40% GC content. Each base is sampled independently.
    """

    def __init__(self, seqLength, discreteDistribution=util.DEFAULT_BASE_DISCRETE_DISTRIBUTION):
        """
            seqLength: the length of the sequence to return. Can also be an instance of AbstractQuantityGenerator
            discreteDistribution: instance of util.DiscreteDistribution
        """
        super(ZeroOrderBackgroundGenerator, self).__init__(
            SampleFromDiscreteDistributionSubstringGenerator(discreteDistribution), seqLength)

###
# Older API below...this was just set up to generate the background sequence
###


def getGenerationOption(string):  # for yaml serialisation
    return util.getFromEnum(GENERATION_OPTION, "GENERATION_OPTION", string)
GENERATION_OPTION = util.enum(zeroOrderMarkov="zrOrdMrkv")


def getFileNamePieceFromOptions(options):
    return options.generationOption + "_seqLen" + str(options.seqLength)


def generateString_zeroOrderMarkov(length, discreteDistribution=util.DEFAULT_BASE_DISCRETE_DISTRIBUTION):
    """
        discreteDistribution: instance of util.DiscreteDistribution
    """
    sampledArr = util.sampleNinstancesFromDiscreteDistribution(
        length, discreteDistribution)
    return "".join(sampledArr)


def generateString(options):
    if options.generationOption == GENERATION_OPTION.zeroOrderMarkov:
        return generateString_zeroOrderMarkov(length=options.seqLength)
    else:
        raise RuntimeError("Unsupported generation option: " +
                           str(options.generationOption))


def getParentArgparse():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--generationOption", default=GENERATION_OPTION.zeroOrderMarkov, choices=GENERATION_OPTION.vals)
    parser.add_argument("--seqLength", type=int, required=True,
                        help="Length of the sequence to generate")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[getParentArgparse()])
    parser.add_argument("--numSamples", type=int, required=True)
    options = parser.parse_args()

    outputFileName = getFileNamePieceFromOptions(
        options) + "_numSamples-" + str(options.numSamples) + ".txt"

    outputFileHandle = open(outputFileName, 'w')
    outputFileHandle.write("id\tsequence\n")
    for i in range(options.numSamples):
        outputFileHandle.write("synthNeg" + str(i) +
                               "\t" + generateString(options) + "\n")
    outputFileHandle.close()
