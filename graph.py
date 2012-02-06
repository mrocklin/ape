

class Node(object):
    node_color='red'
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(self.info())
    def __eq__(self, other):
        return self.info() == other.info()

    def to_network(self, graph=None):
        import networkx as nx
        if not graph:
            graph = nx.DiGraph()
            graph.add_node(self)
        for parent in self.parents:
            if parent in graph:
                continue
            graph.add_node(parent)
            graph.add_edge(self, parent)
            graph = parent.to_network(graph)
        return graph

    def draw(self):
        import networkx as nx
        G = self.to_network()
        nx.draw_networkx(G, node_color=[n.node_color for n in G.nodes()])

