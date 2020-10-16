from __future__ import absolute_import, division, print_function
# String manipulation imports
import re
import string


def normalize(s: str) -> str:
    """Preprocess text

    Numbers to words, punctuation to space
    Args:
        s (str): string to normalize

    Returns:
        str: Normalized string
    """
    regex = re.compile(f'[{re.escape(string.punctuation)}]' )
    s = regex.sub(' ', s)
    s = s.replace(' 0 ',' cero ')
    s = s.replace(' 1 ',' one ')
    s = s.replace(' 2 ',' two ')
    s = s.replace(' 3 ',' three ')
    s = s.replace(' 4 ',' four ')
    s = s.replace(' 5 ',' five ')
    s = s.replace(' 6 ',' six ')
    s = s.replace(' 7 ',' seven ')
    s = s.replace(' 8 ',' eight ')
    s = s.replace(' 9 ',' nine ')
    s = s.lower()
    s = ' '.join(s.split())
    return s