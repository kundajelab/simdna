from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import re
import gzip
import os
import json
from simdna import random
import numpy as np

DEFAULT_LETTER_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


DEFAULT_BACKGROUND_FREQ = OrderedDict(
    [('A', 0.3), ('C', 0.2), ('G', 0.2), ('T', 0.3)])


DEFAULT_DINUC_FREQ = OrderedDict([
 ('AA',0.095),
 ('AC',0.050),
 ('AG',0.071),
 ('AT',0.075),
 ('CA',0.073),
 ('CC',0.054),
 ('CG',0.010),
 ('CT',0.072),
 ('GA',0.060),
 ('GC',0.044),
 ('GG',0.054),
 ('GT',0.050),
 ('TA',0.064),
 ('TC',0.060),
 ('TG',0.073),
 ('TT',0.095),
])


class DiscreteDistribution(object):

    def __init__(self, valToFreq):
        """
            valToFreq: dict where the keys are the possible things to sample, and the values are their frequencies
        """
        self.valToFreq = valToFreq
        self.keysOrder = sorted(valToFreq.keys()) #sort the keys for determinism
        self.freqArr = [valToFreq[key] for key in self.keysOrder]  # array representing only the probabilities
        assert abs(sum(self.freqArr)-1.0) < 10**-5
        # map from index in freqArr to the corresponding value it represents
        self.indexToVal = dict((x[0], x[1]) for x in enumerate(self.keysOrder))

    def sample(self):
        """Sample from the distribution.
        """
        return self.indexToVal[sampleFromProbsArr(self.freqArr)]


DEFAULT_BASE_DISCRETE_DISTRIBUTION = DiscreteDistribution(
    DEFAULT_BACKGROUND_FREQ)


def get_file_handle(filename, mode="r"):
    """
    Retrieve an open file handle
    WARNING: must close file handle returned from this function
    :param filename: str, path to file
    :param mode: char, 'r'=read; 'w'=write, etc according `open`
    :return: open file handle
    """
    if (re.search('.gz$',filename) or re.search('.gzip',filename)):
        if (mode=="r"):
            mode="rb";
        elif (mode=="w"):
            # I think write will actually append if the file already
            # exists...so you want to remove it if it exists
            if os.path.isfile(filename):
                os.remove(filename)
        return gzip.open(filename,mode)
    else:
        return open(filename,mode) 


def default_tab_seppd(s):
    s = trim_newline(s)
    s = s.split("\t")
    return s


def trim_newline(s):
    return s.rstrip('\r\n')


def perform_action_on_each_line_of_file(
    file_handle, action, transformation=default_tab_seppd, ignore_input_title=False):
    """
    Read file and perform action on each line
    :param file_handle: file, file handle
    :param action: function handle, what to do with line
    :param transformation: function handle, manipulate line before action
    :param ignore_input_title: bool, skip index 0
    :return:
    """

    i = 0
    for line in file_handle:
        i += 1
        if hasattr(line, "decode"): 
            line = line.decode("utf-8")
        process_line(line, i, ignore_input_title,
                     transformation, action)

    file_handle.close()


def process_line(line, i, ignore_input_title,
                 transformation, action):
    """
    Line by line file processor; used in motif loading
    and simdata loading functions
    :param line: str, line from file
    :param i: int, line index
    :param ignore_input_title: bool, skip index 0
    :param transformation: function handle, manipulate line before action
    :param action: function handle, what to do with line
    :return:
    """
    if i > 1 or (ignore_input_title is False):
        action(transformation(line), i)


class VariableWrapper():
    """ For when I want reference-type access to an immutable"""
    def __init__(self, var):
        self.var = var   


def enum(**enums):
    """
    Constructs an enum object around a set of kwargs (all of the same length)
    :param enums: dict of iterables of the same length
    :return: enum version of kwargs
    """
    class Enum(object):
        pass
    to_return = Enum
    for key,val in enums.items():
        if hasattr(val, '__call__'): 
            setattr(to_return, key, staticmethod(val))
        else:
            setattr(to_return, key, val)
    to_return.vals = [x for x in enums.values()]
    to_return.the_dict = enums
    return to_return


def combine_enums(*enums):
    new_enum_dict = OrderedDict()
    for an_enum in enums:
        new_enum_dict.update(an_enum.the_dict)
    return enum(**new_enum_dict)


def sampleFromProbsArr(arrWithProbs):
    """Samples from a discrete distribution.

    Arguments:
        arrWithProbs: array of probabilities

    Returns:
        an index, sampled with the probability of that index in
    array of probabilities.
    """
    arrWithProbs = np.array(arrWithProbs)
    return random.choice(len(arrWithProbs), p=arrWithProbs/arrWithProbs.sum())


reverseComplementLookup = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
                           'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'N': 'N', 'n': 'n'}


def reverseComplement(sequence):
    """
    Get the reverse complement of a sequence by flipping
    the pairs of nucleotides and reversing the string
    :param sequence: str, sequence of elements in  reverseComplementLookup
    :return: str, reversed complement
    """
    reversedSequence = sequence[::-1]
    reverseComplemented = "".join(
        [reverseComplementLookup[x] for x in reversedSequence])
    return reverseComplemented


def sampleWithoutReplacement(arr, numToSample):
    arrayCopy = [x for x in arr]
    for i in range(numToSample):
        randomIndex = int(random.random() * (len(arrayCopy) - i)) + i
        swapIndices(arrayCopy, i, randomIndex)
    return arrayCopy[0:numToSample]


def swapIndices(arr, idx1, idx2):
    temp = arr[idx1]
    arr[idx1] = arr[idx2]
    arr[idx2] = temp


def get_file_name_parts(file_name):
    """
    Extract filename components with regex
    :param file_name: a unix file path
    :return:
    """
    p = re.compile(r"^(.*/)?([^\./]+)(\.[^/]*)?$")
    m = p.search(file_name)
    return FileNameParts(m.group(1), m.group(2), m.group(3))


class FileNameParts(object):
    """
    Warning: this will break for non-unix systems;
    wrapper on filename for manipulating file names
    """

    def __init__(self, directory, core_file_name, extension):
        self.directory = directory if (directory is not None) else os.getcwd()
        self.core_file_name = core_file_name
        self.extension = extension

    def get_full_path(self):
        return self.directory+"/"+self.file_name

    def get_core_file_name_and_extension(self):
        return self.core_file_name+self.extension

    def get_transformed_core_file_name(self, transformation, extension=None):
        to_return = transformation(self.core_file_name)
        if (extension is not None):
            to_return = to_return + extension
        else:
            if (self.extension is not None):
                to_return = to_return + self.extension
        return to_return

    def get_transformed_file_path(self, transformation, extension=None):
        return (self.directory+"/"+
                self.get_transformed_core_file_name(transformation,
                                                    extension=extension))


def format_as_json(jsonable_data):
    return json.dumps(jsonable_data, indent=4, separators=(',', ': '))


class ArgumentToAdd(object):
    """
        Class to append runtime arguments to a string
        to facilitate auto-generation of output file names.
    """
    def __init__(self, val, argumentName=None, argNameAndValSep="-"):
        self.val = val;
        self.argumentName = argumentName;
        self.argNameAndValSep = argNameAndValSep;
    def argNamePrefix(self):
        return ("" if self.argumentName is None else self.argumentName+str(self.argNameAndValSep))
    def transform(self):
        string = (','.join([str(el) for el in self.val])\
                   if (isinstance(self.val, str)==False and
                       hasattr(self.val,"__len__")) else str(self.val))
        return self.argNamePrefix()+string;
        # return self.argNamePrefix()+str(self.val).replace(".","p");


class FloatArgument(ArgumentToAdd):
    """
       Replace the period with a p 
    """
    def __init__(self, val, argumentName=None, argNameAndValSep="-"):
        self.val = val;
        self.argumentName = argumentName;
        self.argNameAndValSep = argNameAndValSep;
    def argNamePrefix(self):
        return ("" if self.argumentName is None else self.argumentName+str(self.argNameAndValSep))
    def transform(self):
        string = str(self.val)
        string = string.replace(".","p")
        return self.argNamePrefix()+string


class BooleanArgument(ArgumentToAdd):

    def transform(self):
        assert self.val  # should be True if you're calling transformation
        return self.argumentName


class CoreFileNameArgument(ArgumentToAdd):

    def transform(self):
        import fileProcessing as fp
        return self.argNamePrefix() + fp.getCoreFileName(self.val)


class ArrArgument(ArgumentToAdd):

    def __init__(self, val, argumentName, sep="+", toStringFunc=str):
        super(ArrArgument, self).__init__(val, argumentName)
        self.sep = sep
        self.toStringFunc = toStringFunc

    def transform(self):
        return self.argNamePrefix() + self.sep.join([self.toStringFunc(x) for x in self.val])


class ArrOfFileNamesArgument(ArrArgument):

    def __init__(self, val, argumentName, sep="+"):
        import fileProcessing as fp
        super(ArrOfFileNamesArgument, self).__init__(val, argumentName,
                                                     sep, toStringFunc=lambda x: fp.getCoreFileName(x))


def addArguments(string, args, joiner="_"):
    """
        args is an array of ArgumentToAdd.
    """
    for arg in args:
        string = string + ("" if arg.val is None or arg.val is False or (hasattr(
            arg.val, "__len__") and len(arg.val) == 0) else joiner + arg.transform())
    return string
