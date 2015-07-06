# Fuzzy Multilevel Graph Embedding
Implementation of the Fuzzy Multilevel Graph Embedding Algorithm (http://www.sciencedirect.com/science/article/pii/S0031320312003470). 

####Please note that is not an official implementation of the algorithm. I am in no way related to the authors.

Usage:

- call train(list(networkx.graph)) to learn the fuzzy overlapping
intervals for the graph's attributes (Unsupervised learning phase)
from the learning set
- call embed(networkx.graph) to embed the graph into a numpy.array (vector)
- call embed_list(list(networkx.graph) to embed a list of graphs into a numpy.array

Input Data Format:

- Input data are undirected graphs represented as networkx.graph
- The node and edge attributes can be strings or floats/ints. 
- No two attributes can have the same name! This is also true for attributes from nodes and edges!

Differences to the paper:

I use median instead of mean to calculate the ressemblance
for nodes: In the original paper the authors computed
the ressemblance between all pairs of edges connected to the nodes
then used their mean as the value for the ressemblance for that
attribute. I use the mean (search for line np.median(all_ressemblances) to change)

----

Muhammad Muzzamil Luqman, Jean-Yves Ramel, Josep Lladós, Thierry Brouard, Fuzzy multilevel graph embedding, Pattern Recognition, Volume 46, Issue 2, February 2013, Pages 551-565, ISSN 0031-3203, http://dx.doi.org/10.1016/j.patcog.2012.07.029.
(http://www.sciencedirect.com/science/article/pii/S0031320312003470)

Abstract:

Structural pattern recognition approaches offer the most expressive, convenient, powerful but computational expensive representations of underlying relational information. To benefit from mature, less expensive and efficient state-of-the-art machine learning models of statistical pattern recognition they must be mapped to a low-dimensional vector space. Our method of explicit graph embedding bridges the gap between structural and statistical pattern recognition. We extract the topological, structural and attribute information from a graph and encode numeric details by fuzzy histograms and symbolic details by crisp histograms. The histograms are concatenated to achieve a simple and straightforward embedding of graph into a low-dimensional numeric feature vector. Experimentation on standard public graph datasets shows that our method outperforms the state-of-the-art methods of graph embedding for richly attributed graphs.

