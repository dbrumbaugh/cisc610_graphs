"""
A general utility module containing
implementations of various graph
algorithms from lecture.
"""

import networkx as nx
from matplotlib import pyplot as plt

def example_graph():
    """
    Returns a weighted graph for use in
    examples.

    Positional Parms:
        None

    Keyword Parms:
        None

    Returns:
        G, a NetworkX graph.
    """
    G = nx.Graph()
    G.add_edge(1, 2, weight=.5)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=3)
    G.add_edge(4, 1, weight=1)
    G.add_edge(5, 2, weight=3)
    G.add_edge(5, 3, weight=4)
    G.add_edge(7, 1, weight=2)
    G.add_edge(3, 1, weight=1)
    G.add_edge(8, 2, weight=2)
    G.add_edge(8, 3, weight=.1)
    G.add_edge(9, 5, weight=.6)
    G.add_edge(6, 4, weight=1)

    return G


def display_graph(G, fname=None):
    """
    Display a NetworkX graph using matplotlib

    Positional Parms:
        G -- A NetworkX graph to be displayed
    Keyword Parms:
        fname -- [Optional] A filename to save
                 the graph picture to in the working
                 directory. If not specified, no image
                 is saved.

    Returns:
        None
    """

    pos = nx.spring_layout(G)
    nx.draw(G, nodecolor='r', edge_color='b', pos=pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_labels(G, pos)

    if fname:
        plt.savefig('graph.png')

    plt.show()

def depth_traverse(G, first_node=None):
    """
    An example of the depth-first traversal of a
    graph. This won't search for anything, it will
    only walk through the entire graph.

    Positional Parms:
        G -- A NetworkX graph to be traversed.
    Keyword Parms:
        first_node -- [OPTIONAL] A specified first
                      node from which to begin the
                      traversal. If not specified,
                      the algorithm will use the first
                      node in the graph as converted to
                      a list.
    Returns:
        None
    """
    #Initialize the visited flag to false for all nodes
    for node in G:
        G.nodes[node]['visited'] = False

    #If the starting point is not specified,
    #default to the "first" node in the dict
    if first_node is None:
        first_node = list(G)[0]

    #initialize stack and push first node
    #into it
    node_stack = []
    G.nodes[first_node]['visited'] = True
    node_stack.append(first_node)

    #process nodes on the stack
    while node_stack:
        #get node from stack
        node = node_stack.pop()

        #get adjacent nodes, and process them
        #if not already visited
        for adj_nd in G[node]:
            if G.nodes[adj_nd]['visited']:
                continue

            print('{}\t--->\t{}'.format(node, adj_nd))
            #mark visited and push to stack so that its
            #adjacent nodes can be processed.
            G.nodes[adj_nd]['visited'] = True
            node_stack.append(adj_nd)

from queue import Queue
def breadth_traverse(G, first_node=None):
    """
    An example of the breadth-first traversal of a
    graph. This won't search for anything, it will
    only walk through the entire graph.

    Positional Parms:
        G -- A NetworkX graph to be traversed.
    Keyword Parms:
        first_node -- [OPTIONAL] A specified first
                      node from which to begin the
                      traversal. If not specified,
                      the algorithm will use the first
                      node in the graph as converted to
                      a list.
    Returns:
        None
    """
    #Initialize the visited flag to false for all nodes
    for node in G:
        G.nodes[node]['visited'] = False

    #If the starting point is not specified,
    #default to the "first" node in the dict
    if first_node is None:
        first_node = list(G)[0]

    #initialize queue and push first node
    #into it
    node_queue = Queue()
    G.nodes[first_node]['visited'] = True
    node_queue.put(first_node)

    #process nodes on the queue
    while not node_queue.empty():
        #get node from queue
        node = node_queue.get()

        #get adjacent nodes, and process them
        #if not already visited
        for adj_nd in G[node]:
            if G.nodes[adj_nd]['visited']:
                continue

            print('{}\t--->\t{}'.format(node, adj_nd))
            #mark visited and push to queue so that its
            #adjacent nodes can be processed.
            G.nodes[adj_nd]['visited'] = True
            node_queue.put(adj_nd)


def prim(G, root_nd=None, verbose=False):
    """
    An implementation of Prim's algorithm for calculating the
    minimum spanning tree of a graph.

    Positional Parms:
        G -- a weighted NetworkX graph from which the MST is to
             be determined.
    Keyword Parms:
        root_nd -- The node to use as the first root of the MST.
                   If not specified, use the first node in the graph
                   as converted to a list.
        verbose -- If True, display an image of the tree after each
                   node is added.
    Returns:
        T, the minimum spanning tree of G, rooted at root_nd.
    """
    #Initialize empty tree
    T = nx.Graph()

    #Define "infinity". This is used as the distance between
    #unconnected nodes, and should be larger than the largest
    #weight in the graph.
    infty = 1000

    #Initialize the min_weight and best_edge values
    #for each node.
    for node in G.nodes:
        G.nodes[node]['min_weight'] = infty
        G.nodes[node]['best_edge'] = None

    #Create the list of nodes to be added to the 
    #tree, and set the min_weight of the designated
    #"root" node to 0, so it is the first selected.
    Q = list(G.nodes)
    if root_nd is None:
        root_nd = Q[0]

    G.nodes[root_nd]['min_weight'] = 0

    #Iterate over Q until it is empty.
    #NOTE: This will result in an infinite loop
    #in the case of a graph that is not fully connected
    while Q:
        #Find the node to add to the tree based on
        #the min_weight parameter. Save the index
        #so we can delete it later.
        temp_min = infty
        best_node = None
        best_idx = None
        for i, node in enumerate(Q):
            if G.nodes[node]['min_weight'] < temp_min:
                temp_min = G.nodes[node]['min_weight']
                best_node = node
                best_idx = i

        #Add the node to the tree, along with its best edge
        T.add_node(best_node)
        if G.nodes[best_node]['best_edge'] is None:
            pass
        else:
            a, b = G.nodes[best_node]['best_edge']
            T.add_edge(a, b, weight=temp_min)

        #For each node adjacent to the newly added one,
        #but not yet in the tree, recalculate the best
        #edges and minimum weights.
        for adj_node in G[best_node]:
            weight = G[best_node][adj_node]['weight']
            if weight < G.nodes[adj_node]['min_weight'] and adj_node in Q:
                G.nodes[adj_node]['min_weight'] = weight
                G.nodes[adj_node]['best_edge'] = (best_node, adj_node)

        #If verbose is set, show step-by step processing
        if verbose:
            for node in Q:
                print('{}\t{}'.format(node, G.nodes[node]['min_weight']))
            display_graph(T)

        #Remove the inserted node from Q
        del Q[best_idx]
    #Once loop is done, T will contain the minimum spanning tree.
    return T



def insertion_sort(data):
    """
    Simple implementation of insertion sort, used within
    Kruskal's algorithm. Accepts a collection, data, that
    contains objects with rich comparisions implemented,
    and performs an in-place sort.

    Positional Parms:
        data -- the mutable collection to be sorted, in place.
    Keyword Parms:
        none
    Returns:
        none
    """
    for i in range(1, len(data)):
        j = i
        while j > 0 and data[j][2] < data[j-1][2]:
            data[j], data[j-1] = data[j-1], data[j]
            j = j - 1


def kruskal(G, verbose=False):
    """
    An implementation of Kruskal's algorithm. Accepts a weighted graph,
    G, and returns an associated minimum spanning forest.

    Positional Parms:
        G -- a weighted NetworkX graph to generate the MST from.
    Keyword Parms:
        verbose -- If True, displays an image of the graph each time
                   a new node is added.
    Returns:
        T, the MSF generated from G using Kruskal's Algorithm. If G
        is fully connected, then the MST will be at index 0.
    """
    #Create a forest from the nodes in G
    F = []
    for node in G.nodes:
        F.append(nx.Graph())
        F[-1].add_node(node)

    #Sort the edges of G
    edge_set = list(G.edges.data('weight'))
    insertion_sort(edge_set)

    #Iterate over the edges in G, in ascending weight order
    for a, b, weight in edge_set:
        a_tree = None
        b_tree = None

        #Determine if a and b are in the same tree (linear search)
        for t in F:
            if a in t.nodes:
                a_tree = t
                break

        for t in F:
            if b in t.nodes:
                b_tree = t
                break

        #If so, skip this edge
        if a_tree == b_tree:
            continue

        #If not, merge the trees
        a_tree.add_edge(a, b, weight=weight)
        a_tree.add_edges_from(b_tree.edges)

        if verbose:
            display_graph(a_tree)

        #The b tree has been merged, so delete it from the forest
        del b_tree

    #Return the forest. If graph is connected,
    #then the spanning tree will be at index 0
    return F
