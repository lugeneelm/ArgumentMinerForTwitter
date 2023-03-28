import pydot

# Create an argument visualisation from a dataset of argument pairs and their relations
def plotGraph(relations, pngFilename):
    # separate support and create relations
    support= relations[relations['Relation']==1].reset_index(drop=True)
    conflict= relations[relations['Relation']==0].reset_index(drop=True)

    # Create the directed graph
    graph = pydot.Dot('Argument Graph', graph_type='digraph', layout='dot', suppress_disconnected=True, rankdir='BT',nodesep=0.1, concentrate=True, fontsize="1000pt")
    graph.set_ratio(0.3)

    # Add the tweet pairs as edges to the graph
    # Add support relations as black arrows and conflict as red
    for index, row in support.iterrows():
        graph.add_edge(pydot.Edge(row['Argument1'], row['Argument2'], color='black', weight=2))
    
    for index, row in conflict.iterrows():
        graph.add_edge(pydot.Edge(row['Argument1'], row['Argument2'], color='red', weight=2))

    # Save the graph to a png file
    graph.write_png(pngFilename)
    print("Argument visualisation created")



