from __future__ import absolute_import, division, print_function
import numpy as np
from simdna.simdnautil.util import DEFAULT_LETTER_TO_INDEX
from simdna.simdnautil import util
import math


class PWM(object):
    """
    Object representing a position weight matrix;
    allows sampling from the PWM either randoml or taking the best hit.
    """

    def __init__(self, name, letterToIndex=DEFAULT_LETTER_TO_INDEX):
        self.name = name
        self.letterToIndex = letterToIndex
        self.indexToLetter = dict(
            (self.letterToIndex[x], x) for x in self.letterToIndex)
        self._rows = []
        self._finalised = False


    def add_row(self, weights):
        self.addRow(weights)

    def addRow(self, weights):
        if (len(self._rows) > 0):
            assert len(weights) == len(self._rows[0])
        self._rows.append(weights)

    def add_rows(self, matrix):
        self.addRows(matrix)

    def addRows(self, matrix):
        for row in matrix:
            self.addRow(weights=row)
        return self

    def finalize(self, pseudocountProb=0.001):
        self.finalise(pseudocountProb=pseudocountProb)

    def finalise(self, pseudocountProb=0.001):
        """
        Function run after loading the weight matrix to smooth
        the PWM after loading is complete
        :param pseudocountProb: smoothing factor
        :return:
        """
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
        return self

    def get_best_hit(self):
        return self.bestPwmHit

    def getBestHit(self):
        return self.bestPwmHit

    def compute_best_hit_given_matrix(self, matrix):
        """
        Compute the highest probability instance of the PWM
        :param matrix: the matrix to use to copmute the PWM
        :return: the string best hit
        """
        return "".join(self.indexToLetter[x] for x in (np.argmax(matrix, axis=1)))

    def computeBestHitGivenMatrix(self, matrix):
        """
        Compute the highest probability instance of the PWM
        :param matrix: the matrix to use to copmute the PWM
        :return: the string best hit
        """
        return "".join(self.indexToLetter[x] for x in (np.argmax(matrix, axis=1)))

    def get_rows(self):
        return self.getRows()

    def getRows(self):
        if (not self._finalised):
            raise RuntimeError("Please call finalised on " + str(self.name))
        return self._rows

    def sample_from_pwm(self, bg=None):
        self.sampleFromPwm(bg=bg)

    def sampleFromPwm(self, bg=None):
        """
        Randomly sample according to the PWM; if a background is included
        then compute the logodds relative to that background and return.
        :param bg: background frequency to compute relative to
        :return: sample or (sample and logodds) if bg is not None
        """
        if (not self._finalised):
            raise RuntimeError("Please call finalised on " + str(self.name))

        sampledLetters = []
        logOdds = 0
        for row in self._rows:
            sampledIndex = util.sampleFromProbsArr(row)
            letter = self.indexToLetter[sampledIndex]
            if (bg is not None):
                logOdds += np.log(row[sampledIndex]) - np.log(bg[letter]) 
            sampledLetters.append(letter)
        sampledHit = "".join(sampledLetters)
        if (bg is not None):
            return (sampledHit, logOdds)
        else:
            return sampledHit

    def sample_from_pwm_and_score(self, bg):
        return self.sampleFromPwm(bg=bg)

    def sampleFromPwmAndScore(self, bg):
        return self.sampleFromPwm(bg=bg)

    def __str__(self):
        return self.name + "\n" + str(self._rows)
