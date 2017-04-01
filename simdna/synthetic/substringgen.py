from __future__ import absolute_import, division, print_function
from simdna.synthetic.core import DefaultNameMixin
from simdna import pwm
from simdna import util
import random


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
        return self.pwm.sampleFromPwm(), self.pwm.name

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "PwmSampler"), ("motifName", self.pwm.name)])


class PwmSamplerFromLoadedMotifs(PwmSampler):
    """Instantiates a :class:`.PwmSampler` from an
    :class:`.AbstractLoadedMotifs` object.

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

        name: see :class:`.DefaultNameMixin`
    """

    def __init__(self, pwm, name=None):
        self.pwm = pwm
        super(BestHitPwm, self).__init__(name)

    def generateSubstring(self):
        """See superclass.
        """
        return self.pwm.getBestHit(), self.pwm.name

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "BestHitPwm"), ("pwm", self.pwm.name)])


class BestHitPwmFromLoadedMotifs(BestHitPwm):
    """Instantiates :class:`BestHitPwm` using a :class:`.LoadedMotifs` file.
    Analogous to :class:`.PwmSamplerFromLoadedMotifs`.
    """

    def __init__(self, loadedMotifs, motifName, name=None):
        self.loadedMotifs = loadedMotifs
        super(BestHitPwmFromLoadedMotifs, self).__init__(
            loadedMotifs.getPwm(motifName), name)

    def getJsonableObject(self):
        """See superclass.
        """
        obj = super(BestHitPwmFromLoadedMotifs, self).getJsonableObject()
        return obj
