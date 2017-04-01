from __future__ import absolute_import, division, print_function
import numpy as np
from simdna.util import DEFAULT_LETTER_TO_INDEX
from simdna import util
import math


class PWM(object):

    def __init__(self, name, letterToIndex=DEFAULT_LETTER_TO_INDEX):
        self.name = name
        self.letterToIndex = letterToIndex
        self.indexToLetter = dict(
            (self.letterToIndex[x], x) for x in self.letterToIndex)
        self._rows = []
        self._finalised = False

    def addRow(self, weights):
        if (len(self._rows) > 0):
            assert len(weights) == len(self._rows[0])
        self._rows.append(weights)

    def finalise(self, pseudocountProb=0.001):
        assert pseudocountProb >= 0 and pseudocountProb < 1
        # will smoothen the rows with a pseudocount...
        self._rows = np.array(self._rows)
        self._rows = self._rows * \
            (1 - pseudocountProb) + float(pseudocountProb) / len(self._rows[0])
        for row in self._rows:
            assert(abs(sum(row) - 1.0) < 0.0001)
        self._logRows = np.log(self._rows)
        self._finalised = True
        self.bestPwmHit = self.computeBestHitGivenMatrix(self._rows)
        self.pwmSize = len(self._rows)

    def getBestHit(self):
        return self.bestPwmHit

    def computeBestHitGivenMatrix(self, matrix):
        return "".join(self.indexToLetter[x] for x in (np.argmax(matrix, axis=1)))

    def getRows(self):
        if (not self._finalised):
            raise RuntimeError("Please call finalised on " + str(self.name))
        return self._rows

    def sampleFromPwm(self):
        if (not self._finalised):
            raise RuntimeError("Please call finalised on " + str(self.name))
        sampledLetters = []
        for row in self._rows:
            sampledIndex = util.sampleFromProbsArr(row)
            sampledLetters.append(
                self.indexToLetter[util.sampleFromProbsArr(row)])
        return "".join(sampledLetters)

    def __str__(self):
        return self.name + "\n" + str(self._rows)
