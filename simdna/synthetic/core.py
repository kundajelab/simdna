from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import numpy as np
from simdna import util


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
    ofh = util.get_file_handle(outputFileName, 'w')
    if (includeFasta):
        fastaOfh = util.get_file_handle(util.get_file_name_parts(
            outputFileName).get_transformed_file_path(
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
    infoFilePath = (util.get_file_name_parts(outputFileName)
                        .get_transformed_file_path(
                          lambda x: x + "_info", extension=".txt"))

    ofh = util.get_file_handle(infoFilePath, 'w')
    ofh.write(util.format_as_json(sequenceSetGenerator.getJsonableObject()))
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
