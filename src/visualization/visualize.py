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

from graphviz import Digraph

def ann_viz(model, cache, weights, biases, grads, sample_id, view=True, filename="network.gv",
                         title="My Neural Network", format='svg'):
    nn_per_layer = model.n_neurons_per_layer

    g = Digraph('ANN', filename=filename, format=format)
    maxi = max(nn_per_layer)
    g.graph_attr.update(splines="false", nodesep=str(maxi/2), ranksep=str(2*maxi/3))

    last_layer_nodes = None

    for i in range(0, len(nn_per_layer)):
        layer_act_func = model.layers[i].act_func
        labeljust = 'right'
        node_color = '#3498db'
        the_label = ""
        labelloc = 'b'

        if i == 0:  # input layer
            the_label = title + '\n\n\n\nInput Layer'
            node_color = '#2ecc71'
            labeljust = None
            labelloc = None

        if i == len(nn_per_layer) - 1:  # output layer
            node_color = '#e74c3c'
            labeljust = '1'
            the_label = 'Output Layer'

        with g.subgraph(name="cluster_" + str(i + 1)) as c:
            c.attr(color='white')
            c.attr(rank='same')
            c.attr(labeljust=labeljust, labelloc=labelloc, label=the_label)
            for j in range(0, nn_per_layer[i]):
                b = biases[i][j]
                dAcurr = dAprev = db = '-'
                if i != 0:
                    db = grads['db'+str(i)][j][0]
                    dAcurr = grads['dA_curr'+str(i)][j][sample_id]
                    #dAprev = grads['dA_prev'+str(i)][j][sample_id]

                node_input = cache['Z' + str(i)][j][sample_id]
                node_output = cache['A' + str(i)][j][sample_id]
                node_tooltip = f"b: {b}\nact_func: {layer_act_func}\ni: {node_input}\no: {node_output}\n\nbackward pass:\ndb: {db}\ndAcurr: {dAcurr}"

                c.node(str(i) + ',' + str(j), tooltip=node_tooltip,
                       shape="circle", style="filled", color=node_color, fontcolor=node_color)
                if i != 0:
                    for h in range(last_layer_nodes):
                        w = weights[i][j][h]
                        node_before_output = cache['A' + str(i-1)][h][sample_id]
                        dw = grads['dW'+str(i)][j][h]
                        edge_tooltip = f"w: {w}\nbefore: {node_before_output}\nafter: {w*node_before_output}\n\nbackward pass:\ndw: {dw}"
                        g.edge(str(i - 1) + ',' + str(h), str(i) + ',' + str(j), tooltip=edge_tooltip, penwidth='2')
            last_layer_nodes = nn_per_layer[i]

    g.attr(arrowShape="none")
    g.edge_attr.update(arrowhead="none", color="#707070")
    if view == True:
        g.view()
    return g