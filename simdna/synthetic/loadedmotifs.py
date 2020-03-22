from __future__ import absolute_import, division, print_function
from simdna.synthetic.core import DefaultNameMixin
from simdna import pwm
from simdna import util
from collections import OrderedDict


class AbstractLoadedMotifs(object):
    """Class representing loaded PWMs.

    A class that contains instances of ``pwm.PWM`` loaded from a file.
    The pwms can be accessed by name.

    Arguments:
        loadedMotifs: dictionary mapping names of motifs
    to instances of ``pwm.PWM`` 
    """

    def __init__(self, loadedMotifs):
        self.loadedMotifs = loadedMotifs

    def getPwm(self, name):
        """Get a specific PWM.

        Returns:
            The ``pwm.PWM`` instance with the specified name.
        """
        return self.loadedMotifs[name]

    def addMotifs(self, abstractLoadedMotifs):
        """Adds the motifs in abstractLoadedMotifs to this.

        Arguments:
            abstractLoadedMotifs: instance of :class:`.AbstractLoadedMotifs`

        Returns:
            self, as a convenience
        """
        self.loadedMotifs.update(abstractLoadedMotifs.loadedMotifs)
        return self #convenience return


class AbstractLoadedMotifsFromFile(AbstractLoadedMotifs):
    """Class representing loaded PWMs.

    A class that contains instances of ``pwm.PWM`` loaded from a file.
    The pwms can be accessed by name.

    Arguments:
        fileName: string, the path to the file to load

        pseudocountProb: if some of the pwms have 0 probability for\
    some of the positions, will add the specified ``pseudocountProb``\
    to the rows of the pwm and renormalise.
    """

    def __init__(self, fileName,
                       pseudocountProb=0.0):
        self.fileName = fileName
        fileHandle = util.get_file_handle(fileName)
        self.pseudocountProb = pseudocountProb
        self.loadedMotifs = OrderedDict()
        action = self.getReadPwmAction(self.loadedMotifs)
        util.perform_action_on_each_line_of_file(
            file_handle=fileHandle,
            action=action,
            transformation=util.trim_newline
        )
        for pwm in self.loadedMotifs.values():
            pwm.finalise(pseudocountProb=self.pseudocountProb)
        super(AbstractLoadedMotifsFromFile, self).__init__(self.loadedMotifs)

    def getReadPwmAction(self, loadedMotifs):
        """Action performed when each line of the pwm text file is read in.

        This function is to be overridden by a specific implementation.
        It is executed on each line of the file when it is read in, and
        when PWMs are ready they will get inserted into ``loadedMotifs``.

        Arguments:
            loadedMotifs: an ``OrderedDict`` that will be filled with PWMs.
        The keys will be the names of the PWMs and the
        values will be instances of ``pwm.PWM``
        """
        raise NotImplementedError()


class LoadedEncodeMotifs(AbstractLoadedMotifsFromFile):
    """A class for reading in a motifs file in the ENCODE motifs format.

    This class is specifically for reading files in the encode motif
    format - specifically the motifs.txt file that contains Pouya's motifs
    (http://compbio.mit.edu/encode-motifs/motifs.txt)

    Basically, the motif declarations start with a >, the first
    characters after > until the first space are taken as the motif name,
    the lines after the line with a > have the format:
    "<ignored character> <prob of A> <prob of C> <prob of G> <prob of T>"
    """

    def getReadPwmAction(self, loadedMotifs):
        """See superclass.
        """
        currentPwm = util.VariableWrapper(None)

        def action(inp, lineNumber):
            if (inp.startswith(">")):
                inp = inp.lstrip(">")
                inpArr = inp.split()
                motifName = inpArr[0]
                currentPwm.var = pwm.PWM(motifName)
                loadedMotifs[currentPwm.var.name] = currentPwm.var
            else:
                # assume that it's a line of the pwm
                assert currentPwm.var is not None
                inpArr = inp.split()
                summaryLetter = inpArr[0]
                currentPwm.var.addRow([float(x) for x in inpArr[1:]])
        return action


class LoadedHomerMotifs(AbstractLoadedMotifsFromFile):
    """A class for reading in a motifs file in the Homer motifs format.

    Eg: HOCOMOCOv10_HUMAN_mono_homer_format_0.001.motif in resources
    """

    def getReadPwmAction(self, loadedMotifs):
        """See superclass.
        """
        currentPwm = util.VariableWrapper(None)

        def action(inp, lineNumber):
            if (inp.startswith(">")):
                inp = inp.lstrip(">")
                inpArr = inp.split()
                motifName = inpArr[1]
                currentPwm.var = pwm.PWM(motifName)
                loadedMotifs[currentPwm.var.name] = currentPwm.var
            else:
                # assume that it's a line of the pwm
                assert currentPwm.var is not None
                inpArr = inp.split()
                currentPwm.var.addRow([float(x) for x in inpArr[0:]])
        return action
