# Intro

* Trees are fundamental, but problematic when there's recombination
* New inference methods are starting to address this, but data-model & terminology is confused
* Outline of rest of paper

# ARGs and stochastic processes

* Originally Hudson implemented a CwR simulator
* Later Griffiths took a mathematical approach, coining "ARG"
* Explanation of a Griffiths graph & algorithm
* Nod to other graph encodings e.g. GC
* Other uses of stochastic graphs (ASG)
* Link to "phylogenetic networks"
* More on phylogenetic network types
* Topological restrictions in phylogenetic networks

# The Event ARG data structure

* Historical move from ARG as process to ARG as data structure
* Griffiths ARG involves encoding events: an eARG
* Formal definition of an eARG using node annotation
* Problem 1: in node anotated eARGs edge order is significant
* Problem 2: The node annotated eARG is limited to only representing the CwR (e.g. 2 children per node)
* Problem 3: The node annotated eARG encoding can't represent multiple recombinations on a chromosome or GC
* These 3 problems can be solved using edge annotations 

# Genome ARGs

* We suggest an alternative to the Griffiths eARG where nodes are genomes
* This is a gARG - Fig 2 shows illustration embedded in a pedigree
* Formal gARG definition
* Ramification 1) nodes do not have to be events
* Ramification 2) nodes are all of the same type
* Ramification 3) recombination is most sensibly represented using 2 recombination nodes


# Ancestry resolution

(NB: this is not well worked up yet: see https://github.com/tskit-dev/what-is-an-arg-paper/issues/212 for a suggestion)

* Intro (?? this seems not to go anywhere)
* Definition of a sample-resolved gARG
* Fig 3 explanation - how to resolve WRT samples
* Hudson description
* More Hudson description
* Big ARG vs little ARG

# ARG simplification

* Simplifying involves retaining local tree topology by removing nodes or edges
* Unary nodes are important - showcase Fig 4
* 1st step of simplification: removing (super) diamonds
* 2nd step: remove nodes which are unary everywhere
* 3rd step remove nodes from local trees where the node is unary

# ARGs and SPRs

* SMC approximations are often important
* CwR transforms local trees by SPR moves
* Discussion of SPRs in an eARG (incomplete)
* SPRs and unary nodes?

# Evaluation of sampling probabilities

* Fig 5 Little ARG vs big ARG & likelihood of events
*
*

# ARG inference methods

* Review: backwards-in-time vs along-the-genome
* Heuristic methods are faster and do not attempt for maximum parsiomony.
* Importance sampling & MCMC sampling of ARGs

Fig 6: ARGs from KwARG, ARGweaver, Tsinfer, Relate

* ARGInfer

## Discussion/summary