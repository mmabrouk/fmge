"""
Implementation of the Fuzzy Multilevel Graph Embedding algorithm:
Muhammad Muzzamil Luqman, Jean-Yves Ramel, Josep Llados, Thierry Brouard, Fuzzy multilevel graph embedding, Pattern Recognition, Volume 46, Issue 2, February 2013, Pages 551-565, ISSN 0031-3203, http://dx.doi.org/10.1016/j.patcog.2012.07.029.
(http://www.sciencedirect.com/science/article/pii/S0031320312003470)
Structural pattern recognition approaches offer the most expressive, convenient, powerful but computational expensive representations of underlying relational information. To benefit from mature, less expensive and efficient state-of-the-art machine learning models of statistical pattern recognition they must be mapped to a low-dimensional vector space. Our method of explicit graph embedding bridges the gap between structural and statistical pattern recognition. We extract the topological, structural and attribute information from a graph and encode numeric details by fuzzy histograms and symbolic details by crisp histograms. The histograms are concatenated to achieve a simple and straightforward embedding of graph into a low-dimensional numeric feature vector. Experimentation on standard public graph datasets shows that our method outperforms the state-of-the-art methods of graph embedding for richly attributed graphs.
Keywords: Pattern recognition; Graphics recognition; Graph clustering; Graph classification; Explicit graph embedding; Fuzzy logic

This is not the official implementation of the algorithm.
No warranties of correcthood

Usage:
- call train(list(networkx.graph)) to learn the fuzzy overlapping
intervals for the graph's attributes (Unsupervised learning phase)
from the learning set
- call embed(networkx.graph) to embed the graph into a numpy.array (vector)

Input data format:
- Undirected Graph as a networkx.graph
- Node and edge attributes are strings or floats (if your have vectors
  make them in a list of floats)
- No two attributes should have the same name! This is also true for
attributes from nodes and edges!
- All values of discrete attributes should be in the training set

Differences to the paper:
I use median instead of mean to calculate the ressemblance
for nodes: In the original paper the authors computed
the ressemblance between all pairs of edges connected to the nodes
then used their mean as the value for the ressemblance for that
attribute. I use the mean (search for line np.median(all_ressemblances) to change)
"""
__author__ = "Mahmoud Mabrouk"
__email__ = "mahmoud.mabrouk@tu-berlin.de"
__date__ = "18.06.2015"
__version__ = 1

import networkx as nx
from itertools import combinations
import numpy as np
import pdb


class FMGE():

    train_called = False

    def __init__(self):
        self.intervals = {}  # dict(attribute_name -> interval). fuzzy intervals are a list. crisp intervals are dict value->index

    def train(self, training_graphs, n_intervals):
        """
        Learn the fuzzy overlapping intervals for the graph's
        attributes and saves them in the class.
        This function has to be called before being able to embed
        the graphs into vectors
        Args:
            training_graphs: list(networkx.graph) see readme for the
            properties of the graphs
            n_intervals: Number of fuzzy intervals for continuous attribute
        Returns:
            nothing. The learned intervals will be saved in the object
        """

        print "Training FMGE with %s graphs" % (len(training_graphs))
        # We first add the ressemblance attributes to the graphs (see paper)
        augmented_graphs = []
        for graph in training_graphs:
            self._add_degree_as_attribute(graph)
            augmented_graphs.append(self._augment_graph(graph))

        # Extract all attributes from all the augment graphs
        attributes = self._extract_all_attributes(augmented_graphs)

        # Find intervals for each attribute
        for attribute in attributes:  # attributes is a dict(attribute) -> list(values)
            if isinstance(attributes[attribute][0], str):  # Discrete attribute -> crisp intervals
                self.intervals[attribute] = self._get_discrete_attribute_intervals(attributes[attribute])
            else:  # continuous attribute -> fuzzy interval
                self.intervals[attribute] = self._get_fuzzy_attribute_intervals(attributes[attribute], n_intervals)
        self.train_called = True
        print "Training finished"

    def embed_list(self, graphs):
        """
        Embeds a list of graphs. Returns an array of features
        """
        print "Embedding %s graphs" % (len(graphs))
        results = []
        for graph in graphs:
            results.append(self.embed(graph))
        return np.array(results)

    def embed(self, graph):
        """
        Embeds a graph into a vector.
        train() has to have been already first called first.
        Args:
            graph: networkx.graph in the format described in readme
        Returns:
            numpy.vector
        Features:
            |v|
            |e|
            fuzzy histograms for all edge and node attributes
        """
        if not self.train_called:
            raise Exception("Please call train() first to determine the fuzzy \
                             histograms' intervals")
        feature_vector = []
        fv = feature_vector
        fv.append(float(graph.order()))
        fv.append(float(graph.size()))
        for attr in self.intervals:
            if isinstance(self.intervals[attr], dict):  # Discrete attribute. Crisp intervals
                hist = self._calc_attr_discrete_hist(graph, attr, self.intervals[attr])
            else:
                hist = self._calc_attr_fuzzy_hist(graph, attr, self.intervals[attr])
            fv.extend(hist)
        return np.array(fv)

    def _calc_attr_discrete_hist(self, G, attr, crisp_intevals):
        """
        Compute the crip histogram for a discrete attribute. Returns
        the number of elements in each bin
        Args:
            G: nx.Graph
            attr: Attribute name
            crisp_intevals: dict(value->index)
        Returns:
            list(elem_per_bin)
        """
        ci = crisp_intevals
        vals = nx.get_edge_attributes(G, attr) if nx.get_edge_attributes(G, attr) else nx.get_node_attributes(G, attr)
        vec = [0.0] * len(ci)
        for elem in vals:
            val = vals[elem]
            vec[ci[val]] += 1.0
        return vec

    def _calc_attr_fuzzy_hist(self, G, attr, fuzzy_intervals):
        """
        Computes the fuzzy histogram for an attribute. Returns the
        number of elements in  each bin
        Args:
            G: nx.Graph
            attr: Attribute name
            fuzzy_intervals: list(list(a,b,c,d)) defining a traperzoidal fuzzy histogram
        Returns:
            list(elem_per_bin)
        """
        fi = fuzzy_intervals
        vals = nx.get_edge_attributes(G, attr) if nx.get_edge_attributes(G, attr) else nx.get_node_attributes(G, attr)
        vec = [0.0]*len(fi)
        for elem in vals:
            val = vals[elem]
            for i, interval in enumerate(fi):
                if (val >= interval[0]) and (val < interval[1]):
                    vec[i] += (val-interval[0]) / (interval[1]-interval[0])
                elif (val >= interval[1]) and (val <= interval[2]):
                    vec[i] += 1
                elif (val > interval[2]) and (val <= interval[3]):
                    vec[i] += (val-interval[3]) / (interval[2]-interval[3])
                else:
                    pass
        return vec

    def _add_degree_as_attribute(self, G):
        """
        Adds the degree of each node as an attribute of that node
        Changes the graph object!
        """
        nx.set_node_attributes(G, "_degree", G.degree())

    def _augment_graph(self, G):
        """
        Add the ressemblance attributes (see paper) to a graph
        Args:
            G: networkx.graph (see Readme for properties)
        Returns:
            augmented graph
        """
        node_attributes = G.node[G.nodes()[0]].keys() if G.nodes() else []
        edge_attributes = G.edge[G.edges()[0][0]][G.edges()[0][1]].keys() if G.edges() else []

        #Augmenting edges
        for attr in node_attributes:
            self.l._debug("Node attribute %s" % attr)
            attrs = nx.get_node_attributes(G, attr)  # dict node->attribute
            ressemblance = {}
            for edge in G.edges():
                ressemblance[edge] = self.__ressemblance(attrs[edge[0]], attrs[edge[1]])
            nx.set_edge_attributes(G, "ressemblance_" + str(attr), ressemblance)

        # Augmenting nodes
        for attr in edge_attributes:
            self.l._debug("Edge attribute %s" % attr)
            if isinstance(attr, str) and len(attr) > 13:
                if attr[:12] == "ressemblance":
                    self.l._debug("Skipping")
                    continue
            attrs = nx.get_edge_attributes(G, attr)
            ressemblance = {}
            for node in G.nodes():
                # Compute the median of ressemblances between all edge pairs going through the node
                all_ressemblances = []
                for edge_1, edge_2 in combinations(G.edges(node), 2):
                    attr_1 = attrs[edge_1] if edge_1 in attrs else attrs[(edge_1[1], edge_1[0])]
                    attr_2 = attrs[edge_2] if edge_2 in attrs else attrs[(edge_2[1], edge_2[0])]
                    _ = self.__ressemblance(attr_1, attr_2)
                    all_ressemblances.append(_)
                ressemblance[node] = np.median(all_ressemblances)
            nx.set_node_attributes(G, "ressemblance_" + str(attr), ressemblance)
        return G

    def __ressemblance(self, attr_1, attr_2):
        """
        Compute the ressemblance (see paper for function)
        Args:
            attr_1 and attr_2: either both floats/int or both strings
        Returns:
            float
        """
        if isinstance(attr_1, str) and isinstance(attr_2, str):
            if attr_1 == attr_2:
                return 1.0
            else:
                return 0.0
        elif (isinstance(attr_1, float) and isinstance(attr_2, float)) or (isinstance(attr_1, int) and isinstance(attr_2, int)):
            # The next line is the current bottleneck of the algorithm
            attr_1 = float(attr_1)
            attr_2 = float(attr_2)
            _ = np.min([np.abs(attr_1), np.abs(attr_2)]) / np.max([np.abs(attr_1), np.abs(attr_2)])
            if np.isnan(_):
                return 1.0  # In case both attributes equal zero
            else:
                return _
        else:
            pdb.set_trace()
            raise Exception("Attributes input to __ressemblance should be either strings or floats!")

    def _extract_all_attributes(self, graphs):
        """
        Extracts the labels of all the attributes in graphs
        Args:
            graph networkx.graph
        Returns dict("attr") -> list(attributes)
        """
        G = graphs[0]  # We first parse the attribute names from the first graph. It should have at least an edge and a node
        node_attr_names = G.node[G.nodes()[0]].keys()
        edge_attr_names = G.edge[G.edges()[0][0]][G.edges()[0][1]].keys()
        if list(set(node_attr_names) & set(edge_attr_names)):  # If there are attributes with the same name from nodes and edges
            raise Exception("Some attributes for nodes and edges have the same names! This is not allowed, please solve the problem and restart.")
        attr = {attr_name : [] for attr_name in (node_attr_names + edge_attr_names)}

        for G in graphs:
            for node_attr in node_attr_names:
                attr[node_attr].extend(nx.get_node_attributes(G, node_attr).values())
            for edge_attr in edge_attr_names:
                attr[edge_attr].extend(nx.get_edge_attributes(G, edge_attr).values())
        return attr

    def _get_fuzzy_attribute_intervals(self, attributes, n):
        """
        Gets the attribute intervals. Implementation of
        GetFuzzyOverlapTrapzInterval in the paper.
        Args:
            attributes: list(float or strings)
            n: number of fuzzy intervals
        Returns:
            a list of quatriples describing the fuzzy intervals.
            The four values describe each traperzoidal interval.
        """
        N = 2 * n - 1
        if len(set([_ for _ in attributes if np.isfinite(_)])) == 1:  # Only one value
            sole_val = [_ for _ in attributes if np.isfinite(_)][0]
            return [[-np.Inf, -np.Inf, sole_val, sole_val],
                    [sole_val, sole_val, np.Inf, np.Inf]]
        crisp_intevals = self.__get_init_crisp_interval(attributes, N)
        fuzzy_intervals = [[None, None, None, None] for _ in range(n)]
        fi = fuzzy_intervals
        ci = crisp_intevals
        fi[0][0] = -np.Inf
        fi[0][1] = -np.Inf
        fi[0][2] = ci[0][1]
        fi[0][3] = ci[1][1]

        j_crisp = 0
        for j in range(1, n):
            j_crisp += 2
            fi[j][0] = fi[j-1][2]
            fi[j][1] = fi[j-1][3]
            if j_crisp + 1 >= N:
                fi[j][2] = np.Inf
                fi[j][3] = np.Inf
            else:
                fi[j][2] = ci[j_crisp+1][1]
                fi[j][3] = ci[j_crisp+2][1]

        return fi

    def _get_discrete_attribute_intervals(self, attributes):
        """
        Creates a histogram for the discrete values of an attribute.
        Args:
            attributes: list(string)
        Returns:
            dict (value -> histogram index)
        """
        return {_: i for (i, _) in enumerate(list(set(attributes)))}  # python magic

    def __get_init_crisp_interval(self, attributes, n):
        """
        Create equally spaced intervals for the attributes
        Args:
            attributes: list(float)
            n: number of intervals to create
        Returns:
            list of tuples describing the start and end of the
            the intervals
        """
        try:
            equally_spaced_bins = np.arange(np.nanmin(attributes), np.nanmax(attributes), (np.nanmax(attributes) - np.nanmin(attributes)) / float(n+1))
            esb = equally_spaced_bins
            crisp_intevals = [[None, None] for _ in range(n)]
            ci = crisp_intevals
            st = - np.Inf
            en = esb[0]
            for j in range(0, n):
                ci[j][0] = st
                ci[j][1] = en
                st = en
                en = esb[j+1]
        except:
            pdb.set_trace()
        return crisp_intevals
