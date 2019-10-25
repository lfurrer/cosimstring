#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Retrieve similar strings based on cosine distance of s-grams.
"""


from __future__ import unicode_literals


import itertools as it
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix


__version__ = '1.0.0'


# Py2/3 compatibility
# Text types: (str, unicode) for Py2, (str,) for Py3.
TEXT_TYPES = tuple(set([str, ''.__class__]))
# Case folding.
try:
    casefold = str.casefold  # Py3: str.lower plus a few extra conversions
except AttributeError:
    casefold = lambda text: text.lower()  # Py2: str.lower and unicode.lower


class Collection(object):
    """Container for a collection of strings to retrieve from."""

    DTYPE = np.float32
    PAD_SYMBOL = '\x00'

    def __init__(self, collection, sgrams=(2, 3), pad='both', norm=casefold):
        self.shapes = tuple(self._parse_shapes(sgrams))
        self.pad = self._pad_template(pad)
        self.norm = norm

        self._coll = list(collection)
        self._sgram_index = {}  # s-gram vocabulary
        self._sgram_matrix = self._create_matrix(self._coll, update=True).T

    def most_similar(self, query, threshold=.7, cutoff=25, sort=False):
        """
        Get the most similar items from the collection.

        Compare all strings in `query` against the collection
        and yield a list of similar collection items for each
        query item.
        If query is a single string, return a single list.

        The result lists contain only strings with a similarity
        of `threshold` or higher.
        The result lists contain at most `cutoff` items.
        If `sort` is True, the items are sorted by similarity
        in descending order.
        """
        return self.retrieve(query).most_similar(threshold, cutoff, sort)

    def scores(self, query, threshold=.7, cutoff=25, sort=False):
        """
        Get pairs <text, score> for the most similar collection items.

        The parameters `query`, `threshold`, `cutoff` and `sort`
        are interpreted the same ways as for most_similar().
        """
        return self.retrieve(query).scores(threshold, cutoff, sort)

    def retrieve(self, query):
        """
        Get a container with precomputed similarity.

        Use the returned Similarity object to test different
        threshold/cutoff combinations over the same pair of
        query and collection.
        """
        single = False
        if isinstance(query, TEXT_TYPES):
            query = [query]
            single = True
        sgrams = self._create_matrix(query)
        sim = sgrams.dot(self._sgram_matrix)
        sim = sim.toarray()
        return Similarity(sim, self._coll, single)

    def _create_matrix(self, coll, update=False):
        data_triple = self._matrix_data(coll, update)
        rows = len(data_triple[2]) - 1     # len(indptr) == rows+1
        cols = len(self._sgram_index) + 1  # add a column for unseen s-grams
        return csr_matrix(data_triple, shape=(rows, cols))

    def _matrix_data(self, coll, update):
        """Create a <data, indices, indptr> triple for a CSR matrix."""
        data, indices, indptr = [], [], [0]
        vocabulary = self._sgram_index
        lookup = vocabulary.setdefault if update else vocabulary.get
        for text in coll:
            sgrams = Counter(self._preprocess(text))
            indices.extend(lookup(s, len(vocabulary)) for s in sgrams)
            indptr.append(len(indices))
            row = np.fromiter(sgrams.values(), count=len(sgrams),
                              dtype=self.DTYPE)
            row /= np.sqrt(np.square(row).sum())  # L2-normalise
            data.append(row)
        return np.concatenate(data), indices, indptr

    def _preprocess(self, text):
        text = self.norm(text)
        return it.chain.from_iterable(self._skipgrams(text, n, k)
                                      for n, k in self.shapes)

    def _skipgrams(self, text, n, k):
        if self.pad:
            text = self.pad.format(text=text, pad=self.PAD_SYMBOL*(n-1))
        for i in range(len(text)-n+1):
            head = (text[i],)
            for tail in it.combinations(text[i+1 : i+n+k], n-1):
                yield head + tail

    @staticmethod
    def _parse_shapes(shapes):
        for shape in shapes:
            try:
                n, k = shape
            except (ValueError, TypeError):
                n, k = shape, 0
            yield int(n), int(k)

    @staticmethod
    def _pad_template(scheme):
        if scheme == 'none':
            return None
        if scheme == 'left':
            return '{pad}{text}'
        if scheme == 'right':
            return '{text}{pad}'
        if scheme == 'both':
            return '{pad}{text}{pad}'
        raise ValueError('invalid shape: {}'.format(scheme))


class Similarity(object):
    """Container for precomputed similarity between query and collection."""

    def __init__(self, matrix, collection, single=False):
        self.matrix = matrix
        self._coll = collection
        self._return = next if single else iter

    def most_similar(self, threshold=.7, cutoff=25, sort=False):
        """Get the most similar items from the collection."""
        return self._return(self._most_similar(threshold, cutoff, sort))

    def scores(self, threshold=.7, cutoff=25, sort=False):
        """Get pairs <text, score> for the most similar collection items."""
        return self._return(self._scores(threshold, cutoff, sort))

    def _most_similar(self, threshold, cutoff, sort):
        for ind in self._indices(threshold, cutoff, sort):
            yield [self._coll[i] for i in ind]

    def _scores(self, threshold, cutoff, sort):
        for n, ind in enumerate(self._indices(threshold, cutoff, sort)):
            yield [(self._coll[i], self.matrix[n, i]) for i in ind]

    def _indices(self, threshold, cutoff, sort):
        if threshold is None:
            threshold = -1.
        if cutoff is None or cutoff < 0:
            cutoff = self.matrix.shape[1]
        for row in self.matrix:
            yield self._extract(row, threshold, cutoff, sort)

    @staticmethod
    def _extract(row, threshold, cutoff, sort):
        above_threshold = (row >= threshold).sum()
        cutoff = min(cutoff, above_threshold)
        if cutoff == 0:
            return ()  # short-circuit when nothing matches
        kth = range(cutoff) if sort else cutoff-1
        return np.argpartition(-row, kth)[:cutoff]
