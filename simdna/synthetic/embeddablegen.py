from __future__ import absolute_import, division, print_function
from simdna.synthetic.core import DefaultNameMixin
from simdna.synthetic.embeddables import StringEmbeddable
from simdna.synthetic.substringgen import AbstractSubstringGenerator
from simdna.synthetic.embeddables import PairEmbeddable
from collections import OrderedDict


class AbstractEmbeddableGenerator(DefaultNameMixin):
    """Generates an embeddable, usually for embedding in a background sequence.
    """

    def generate_embeddable(self):
        self.generateEmbeddable()

    def generateEmbeddable(self):
        """Generate an embeddable object.

        Returns:
            An instance of :class:`AbstractEmbeddable`
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
        AbstractEmbeddableGenerator.__init__(self, name)

    def generateEmbeddable(self):
        substring, substringDescription =\
            self.substringGenerator.generateSubstring()
        return StringEmbeddable(substring, substringDescription)

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([("class", "SubstringEmbeddableGenerator"),
    ("substringGenerator", self.substringGenerator.getJsonableObject())])
