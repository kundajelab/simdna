from __future__ import absolute_import, division, print_function
from simdna.synthetic.core import DefaultNameMixin
from simdna import random
from collections import OrderedDict
import math

class AbstractPositionGenerator(DefaultNameMixin):
    """Generate a start position at which to embed something

    Given the length of the background sequence and the length
    of the substring you are trying to embed, will return a start position
    to embed the substring at.
    """

    def generate_pos(self, lenBackground, lenSubstring, additionalInfo=None):
        self.generatePos(lenBackground, lenSubstring, additionalInfo=additionalInfo)

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


class NormalDistributionPositionGenerator(AbstractPositionGenerator):
    """Generate position according to normal distribution with mean at
    offsetFromCenter
    """

    def __init__(self, stdInBp, offsetFromCenter=0, name=None):
        super(NormalDistributionPositionGenerator, self).__init__(name)
        self.stdInBp = stdInBp
        self.offsetFromCenter = offsetFromCenter


    def _generatePos(self, lenBackground, lenSubstring, additionalInfo):
        from scipy.stats import norm
        center = (lenBackground-lenSubstring)/2.0
        validPos = False
        totalTries = 0
        while (validPos == False):
            sampledPos = int(norm.rvs(loc=center+self.offsetFromCenter,
                          scale=self.stdInBp))
            totalTries += 1
            if (sampledPos > 0 and sampledPos < (lenBackground-lenSubstring)):
                validPos = True
            if (totalTries%10 == 0 and totalTries > 0):
                print("Warning: made "+str(totalTries)+" attempts at sampling"
                      +" a position with lenBackground "+str(lenBackground)
                      +" and center "+str(center)+" and offset "
                      +str(self.offsetFromCenter)) 
        return sampledPos

    def getJsonableObject(self):
        """Get JSON object representation.

        Returns:
            A json-friendly object (built of dictionaries, lists and
        python primitives), which can be converted to json to
        record the exact details of what was simualted.
        """
        return OrderedDict([("class", "NormalDistributionPositionGenerator"),
                            ("stdInBp", self.stdInBp),
                            ("offsetFromCenter", self.offsetFromCenter)])


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


class FixedPositionGenerator(AbstractPositionGenerator):
    """Sample position uniformly at random.

    Samples a start position to embed the substring in uniformly at random;
        does not return positions that are too close to the end of the
        background sequence to embed the full substring.

    Arguments:
        name: string, see :class:`.DefaultNameMixin`
    """

    def __init__(self, pos, name=None):
        super(FixedPositionGenerator, self).__init__(name)
        self.pos = abs(pos)

    def _generatePos(self, lenBackground, lenSubstring, additionalInfo):
        assert (self.pos < lenBackground - lenSubstring)
        return self.pos

    def getJsonableObject(self):
        """See superclass.
        """
        return "fixed" + str(self.pos)

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
