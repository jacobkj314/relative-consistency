#from https://github.com/jacobkj314/relative-consistency

from math import comb

def min_c(n,a):
    '''
    The minimum possible number of consistent pairs (consistency), out of n total pairs, given the number of accurate instances

    Parameters:
    n (int): the total number of pairs in the dataset
    a (int): the number of accurate instances (accuracy)

    Returns:
    int: the minimum number of consistent pairs (consistency)
    '''
    if a <= n:
        return 0
    return a-n
def max_c(a):
    '''
    The maximum possible number of consistent pairs (consistency), out of n total pairs, given the number of accurate instances

    Parameters:
    a (int): the number of accurate instances (accuracy)

    Returns:
    int: the maximum number of consistent pairs (consistency)
    '''
    return a // 2

def total_mass(n,a):
    '''
    The number of ways to achieve, out of n total pairs, a accurate instances

    Parameters:
    n (int): the total number of pairs in the dataset
    a (int): the number of accurate instances (accuracy)

    Returns:
    int: the number of ways to achieve, out of n total pairs, a accurate instances
    '''
    return comb(2*n, a)
def mass(n,a,c):
    '''
    The number of ways to achieve c consistent pairs, out of n total pairs, given a accurate instances

    Parameters
    n (int): the total number of pairs in the dataset
    a (int): the number of accurate instances (accuracy)
    c (int): the number of consistent pairs (consistency), must be in the interval [min_c(n,a), max_c(n,a)]

    Returns:
    int: the number of ways to achieve c consistent pairs, out of n total pairs, given a accurate instances
    '''
    if not (min_c(n,a) <= c <= max_c(a)):
        return 0
    return comb(n,c) * comb(n-c, a - 2*c) * (1 << (a - 2*c))

def relative_consistency(n,a,c):
    '''
    The Relative Consistency score of a model's performance
    Intuitively, this approximates the probability that a model with the same accuracy will achieve a better consistency than this model
    '''
    return sum(mass(n,a,i) for i in range(c+1)) / total_mass(n,a)