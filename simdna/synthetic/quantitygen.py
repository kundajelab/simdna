from __future__ import absolute_import, division, print_function
from simdna.synthetic.core import DefaultNameMixin
from simdna import random
import numpy as np
from collections import OrderedDict


class AbstractQuantityGenerator(DefaultNameMixin):
    """Class for sampling values from a distribution.
    """

    def generate_quantity(self):
        self.generateQuantity()

    def generateQuantity(self):
        """Sample a quantity from a distribution.

        Returns:
            The sampled value.
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
        AbstractQuantityGenerator.__init__(self, name)

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
        return random.poisson(self.mean)

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
        return 1 if (random.random() <= self.prob) else 0

    def getJsonableObject(self):
        """See superclass.
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
            if (self.theMin is None or quantity >= self.theMin) and (self.theMax is None or quantity <= self.theMax):
                return quantity
            if tries % 10 == 0:
                print("warning: made " + str(tries) +
                      " tries at trying to sample from distribution with min/max limits")

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("min", self.theMin),
                            ("max", self.theMax),
                            ("quantityGenerator", self.quantityGenerator.getJsonableObject())])


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
        return OrderedDict([("class", "ZeroInflater"),
                            ("zeroProb", self.zeroProb),
                            ("quantityGenerator", self.quantityGenerator.getJsonableObject())])

