"""
Copyright (C) 2018 by Tudor Gheorghiu

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated
documentation files (the "Software"),
to deal in the Software without restriction,
including without l> imitation the rights to
use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
"""

def ann_viz(model, view=True, filename="network.gv", title="My Neural Network"):
    """Vizualizez a Sequential model.

    # Arguments
        model: A Keras model instance.

        view: whether to display the model after generation.

        filename: where to save the vizualization. (a .gv file)

        title: A title for the graph
    """
    from graphviz import Digraph;
    import json;
    input_layer, hidden_layers, output_layer = model.n_neurons_per_layer[0], model.n_neurons_per_layer[1:-1], \
                                               model.n_neurons_per_layer[-1]
    last_layer_nodes = input_layer;
    nodes_up = input_layer;

    g = Digraph('g', filename=filename, format='svg');
    n = 0;
    g.graph_attr.update(splines="false", nodesep='1', ranksep='2');
    #Input Layer
    with g.subgraph(name='cluster_input') as c:
        the_label = title+'\n\n\n\nInput Layer';
        #if (int(str(model.layers[0].input_shape).split(",")[1][1:-1]) > 10):
        #    the_label += " (+"+str(int(str(model.layers[0].input_shape).split(",")[1][1:-1]) - 10)+")";
        #    input_layer = 10;
        c.attr(color='white')
        for i in range(0, input_layer):
            n += 1;
            c.node(str(n));
        c.attr(label=the_label)
        c.attr(rank='same');
        c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle");
    for i in range(0, len(hidden_layers)):
        with g.subgraph(name="cluster_"+str(i+1)) as c:
            c.attr(color='white');
            c.attr(rank='same');
            #If hidden_layers[i] > 10, dont include all
            the_label = "";
            #if (int(str(model.layers[i].output_shape).split(",")[1][1:-1]) > 10):
            #    the_label += " (+"+str(int(str(model.layers[i].output_shape).split(",")[1][1:-1]) - 10)+")";
            #    hidden_layers[i] = 10;
            c.attr(labeljust="right", labelloc="b", label=the_label);
            for j in range(0, hidden_layers[i]):
                n += 1;
                c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#3498db");
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    g.edge(str(h), str(n));
            last_layer_nodes = hidden_layers[i];
            nodes_up += hidden_layers[i];
    with g.subgraph(name='cluster_output') as c:
        c.attr(color='white')
        c.attr(rank='same');
        c.attr(labeljust="1");
        for i in range(1, output_layer+1):
            n += 1;
            c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#e74c3c");
            for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                g.edge(str(h), str(n));
        c.attr(label='Output Layer', labelloc="bottom")
        c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle");

    g.attr(arrowShape="none");
    g.edge_attr.update(arrowhead="none", color="#707070");
    if view == True:
        g.view();
    return g