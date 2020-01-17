from __future__ import absolute_import, division, print_function
from simdna.util import dinuc_shuffle, util
from simdna.synthetic.substringgen import AbstractSubstringGenerator
from simdna.synthetic.quantitygen import FixedQuantityGenerator, AbstractQuantityGenerator
from collections import OrderedDict


import csv

class AbstractBackgroundGenerator(object):
    """Returns the sequence that :class:`.AbstractEmbeddable` objects
    are to be embedded into.
    """

    def generate_background(self):
        self.generateBackground()

    def generateBackground(self):
        """Returns a sequence that is the background.
        """
        raise NotImplementedError()

    def get_jsonable_object(self):
        self.getJsonableObject()

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
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
            discreteDistribution= util.DiscreteDistribution(
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
        self._priorFrequencies = priorFrequencies
        self._dinucFrequencies = dinucFrequencies
        assert self.seqLength > 0

        #do some sanity checks on dinucFrequencies
        assert abs(sum(dinucFrequencies.values())-1.0) < 10**-7,\
         sum(dinucFrequencies.values())
        assert all(len(key) == 2 for key in dinucFrequencies.keys())

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

    def getJsonableObject(self):
        return OrderedDict([('class', 'FirstOrderBackgroundGenerator'),
                            ('priorFrequencies', self._priorFrequencies),
                            ('dinucFrequencies', self._dinucFrequencies)]
                           )


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


class AbstractShuffler(object):
    """Implements a method to shuffle a supplied sequence"""

    def shuffle(self, string):
        raise NotImplementedError()

    def getJsonableObject(self):
        return OrderedDict([('class', type(self).__name__)])


class DinucleotideShuffler(AbstractShuffler):
    def shuffle(self, string):
        return dinuc_shuffle.dinuc_shuffle(string)  


class BackgroundArrayFromGenerator(AbstractBackgroundGenerator):
    def __init__(self, backgroundGenerator, num_seqs=100):
        """Returns a sequence array from a generator

        Each sequence is sampled independently.

        These sequence arrays can be used to embed motifs in the
        same position in a number of background sequences, useful
        for setting up randomized-background experiments.

        Arguments:
            backgroundGenerator: AbstractBackgroundGenerator, to sample from
            num_seqs: int, number of sequences to be in the returned array
        """
        self.backgroundGenerator = backgroundGenerator
        self.num_seqs = num_seqs

    def generateBackground(self):
        return ["".join(self.backgroundGenerator.generateBackground()) for _ in range(self.num_seqs)]

    def getJsonableObject(self):
        return OrderedDict([('class', 'BackgroundArrayFromGenerator'),
                            ('backgroundGenerator', self.backgroundGenerator.getJsonableObject()),
                            ('num_seqs', self.num_seqs)]
                           )


class BackgroundFromSimData(AbstractBackgroundGenerator):
    def __init__(self, simdata="data/backgrounds.simdata"):
        """
        Cyclically return a backgrounds from a simdata file; on
        reaching the end of the file simply start at the beginning.

        :param simdata: path to simdata file to load
        """
        self.simdata = simdata
        with open(self.simdata) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            next(reader)
            self.seqs = [seq[1] for seq in reader]
        self.idx = 0

    def __len__(self):
        return len(self.seqs)

    def generateBackground(self):
        seq = self.seqs[self.idx]
        self.idx = (self.idx + 1) % len(self)
        return seq

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([
            ("class", "BackgroundFromSimData"),
            ('simdata', self.simdata)]
        )

class BackgroundArrayFromSimData(AbstractBackgroundGenerator):
    def __init__(self, simdata="data/backgrounds.simdata"):
        """
        Returns a sequence array read from a SimData file
        :param simdata: path to simdata file to load
        """
        self.simdata = simdata

    def generateBackground(self):
        """
        This returns the full array of sequences read from the file..
        :return: a list of sequences which can be manipulated in other SimDNA functions
        """
        with open(self.simdata) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            next(reader)
            seqs = [seq[1] for seq in reader]
        return seqs

    def getJsonableObject(self):
        return OrderedDict([('class', 'BackgroundArrayFromSimData'),
                            ('simdata', self.simdata)]
                           )