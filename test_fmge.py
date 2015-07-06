import unittest
import ipdb
import networkx as nx
import numpy as np
from fmge import fmge
import random
import matplotlib.pyplot as plt


class test_FMGE(unittest.TestCase):

    graphs = []

    @classmethod
    def setUpClass(cls):
        cls.graphs = cls._make_rand_graphs(n=100)
        cls.fmge = fmge()

    @classmethod
    def tearUpClass(cls):
        pass

    @classmethod
    def _make_rand_graphs(cls, n=100):
        """
        Generates n attributes random graphs
        Returns:
            list(networkx.graph)
        """
        graphs = []
        n_node_attributes = random.randint(1, 10)
        n_edge_attributes = random.randint(1, 10)
        for i in range(100):
            G = nx.gnm_random_graph(random.randint(2, 30), random.randint(4, 60))
            node_attr = cls._make_attributes(n_node_attributes, G.nodes())
            for attr in node_attr:
                nx.set_node_attributes(G, attr, node_attr[attr])
            edge_attr = cls._make_attributes(n_edge_attributes, G.edges(), prefix="e_")
            for attr in edge_attr:
                nx.set_edge_attributes(G, attr, edge_attr[attr])
            graphs.append(G)
        return graphs

    @classmethod
    def _make_attributes(cls, n, keys, prefix=""):
        """
        Make n attributes
        """
        attr = {}
        for i in range(n):
            node_attr = {}
            for key in keys:
                node_attr[key] = random.gauss(i, i)
            attr[prefix + str(i)] = node_attr
        return attr

    def test_augment_graph(self):
        """
        Checks that a new ressemblance attribute has
        been created for each node and edge attribute
        """
        augmented_graphs = []
        for graph in self.graphs:
            _ = self.fmge._augment_graph(graph)
            augmented_graphs.append(_)
        self.assertTrue(True)

    def test_extract_all_attributes(self):
        attr = self.fmge._extract_all_attributes(self.graphs)
        self.assertTrue(True)

    def test_get_attribute_intervals(self):
        _ = self.fmge._get_fuzzy_attribute_intervals(np.arange(0.1, np.pi, 0.1), 20)

    def test_train(self):
        _ = self.fmge.train(self.graphs, 8)
        for interval in self.fmge.intervals:
            print interval, self.fmge.intervals[interval]
        for G in self.graphs:
            print self.fmge.embed(G)

if __name__ == "__main__":
    unittest.main()
