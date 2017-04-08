from __future__ import absolute_import, division, print_function
from simdna.synthetic.core import DefaultNameMixin
import re


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
