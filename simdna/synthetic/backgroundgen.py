from __future__ import absolute_import, division, print_function
from simdna import dinuc_shuffle
from simdna.synthetic.substringgen import AbstractSubstringGenerator
from simdna import util
from simdna.synthetic.quantitygen import FixedQuantityGenerator
from collections import OrderedDict


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
