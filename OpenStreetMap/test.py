import osmnx as ox
G = ox.graph_from_point((40.7128, -74.3562), dist=1000, network_type='drive')
ox.plot_graph(G)