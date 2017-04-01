from __future__ import absolute_import, division, print_function
from simdna.synthetic.core import DefaultNameMixin
from simdna.synthetic.positiongen import uniformPositionGenerator
from simdna.synthetic.embeddablegen import SubstringEmbeddableGenerator
from simdna.synthetic.quantitygen import AbstractQuantityGenerator
from simdna import util


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
