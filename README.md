# Cosine-based String Similarity: `cosimstring`

## Usage

Basic usage: create a `Collection` and query it:
```pycon
>>> import cosimstring
>>> # A toy collection of words:
>>> c = cosimstring.Collection('word ward worried record'.split())
>>> # Get all collection elements above the default threshold of 0.7:
>>> c.most_similar('records')
['record']
>>> # Sort the entire collection against the given query:
>>> c.scores('words', threshold=None, sort=True)
[('word', 0.66899353),
 ('worried', 0.40360373),
 ('ward', 0.2508726),
 ('record', 0.21483445)]
```

Typical usage: read a (large) collection from an external resource,
query it with multiple items:
```pycon
>>> with open('large-word-list.txt') as f:
...     c = cosimstring.Collection(l.strip() for l in f)

>>> # If the query is an iterable of strings, the return value is an iterator:
>>> queries = ['multiple', 'words', 'for', 'querying']
>>> for query, matches in zip(queries, c.most_similar(queries)):
...     print(query, len(matches))
multiple 9
words 12
for 3
querying 7
```

Advanced usage: modify the representation of the stored strings,
store the similarity values for later querying:
```pycon
>>> c = cosimstring.Collection(
...     collection,                # any iterable of strings
...     sgrams=[(2, 1), (3, 1)],   # skip-grams of order <2,1> and <3,1>
...     pad='none',                # 'right', 'left', 'both', or 'none'
...     norm=PorterStemmer().stem  # any callable that accepts a string
... )
>>> # Store the cosine scores in a Similarity object:
>>> sim = c.retrieve(query)
>>> # Retrieve different subsets using the precomputed cosine scores:
>>> matches = sim.most_similar(threshold=None, cutoff=5)
>>> scored = sim.scores(threshold=.4, sort=True)
```
