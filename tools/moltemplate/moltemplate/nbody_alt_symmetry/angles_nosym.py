try:
    from .nbody_graph_search import Ugraph
except (SystemError, ValueError):
    # not installed as a package
    from nbody_graph_search import Ugraph


# This file defines how 3-body angle interactions are generated by moltemplate
# by default.  It can be overridden by supplying your own custom file.

#    To find 3-body "angle" interactions, we would use this subgraph:
#
#
#       *---*---*           =>  1st bond connects atoms 0 and 1
#       0   1   2               2nd bond connects atoms 1 and 2
#

bond_pattern = Ugraph([(0, 1), (1, 2)])
# (Ugraph atom indices begin at 0, not 1)


#    The next function eliminates the redundancy between 0-1-2 and 2-1-0:
def canonical_order(match):
    """
    When searching for atoms with matching bond patterns GraphMatcher
    often returns redundant results. We must define a "canonical_order"
    function which sorts the atoms and bonds in a way which is consistent
    with the type of N-body interaction being considered.
    However, some angle_styles (such as angle_style class2)
    have no symmetry (at least not for arbitrary choices of parameters).
    These force-field styles, the different permulations of atom-order
    are not equivalent.  So we do not want to rearrange the order of
    the atoms (and bonds) in the match, because the formula for the
    interaction between atoms 1,2,3 is not the same as the formula
    for the interaction between atoms 3,2,1.
    In this case, this function returns
    the original "match" argument unmodified.

    """

    return match