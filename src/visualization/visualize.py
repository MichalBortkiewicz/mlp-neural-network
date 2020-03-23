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
import numpy as np
import matplotlib.pyplot as plt


def value_to_color(val, min_val, max_val, val_min_color, val_max_color, zero_is_white=True):
    val_max_color = np.array(val_max_color)
    val_min_color = np.array(val_min_color)
    white_color = np.array((255, 255, 255))
    new_min_val = -(max(-min_val, max_val))
    new_max_val = -new_min_val
    alpha = scale_value(val, new_min_val, new_max_val)
    #colormap = plt.get_cmap('RdYlBu')
    if zero_is_white:
        zero_alpha = scale_value(0, new_min_val, new_max_val)
        if alpha <= zero_alpha:
            alpha /=zero_alpha
            color = (1 - alpha) * val_min_color + (alpha) * white_color
        elif alpha > zero_alpha:
            alpha = (alpha - zero_alpha)/(1-zero_alpha)
            color = (alpha) * val_max_color + (1 - alpha) * white_color
    else:
        #color = np.array(colormap(alpha)[:-1])*255
        color = (alpha) * val_max_color + (1 - alpha) * val_min_color

    #color = np.array(colormap(alpha)[:-1]) * 255
    color = color.astype(int)
    return '#%02x%02x%02x' % tuple(color)


def scale_value(val, min_val, max_val, min_scaled_val=0, max_scaled_val=1):
    alpha = (val-min_val)/(max_val-min_val)
    scaled_val = alpha*(max_scaled_val-min_scaled_val)+min_scaled_val
    return min(scaled_val, max_scaled_val)


def ann_viz(model, cache, weights, biases, grads, sample_id, view=True, filename="network.gv",
                         title="My Neural Network", format='svg', color_by_value=True, color_by_gradient=False):
    nn_per_layer = model.n_neurons_per_layer

    g = Digraph('ANN', filename=filename, format=format)
    maxi = max(nn_per_layer)
    g.graph_attr.update(splines="false", nodesep=str(maxi/2), ranksep=str(2*maxi/3))

    last_layer_nodes = None

    NODE_MAX_VAL_COLOR = (0, 0, 255)
    NODE_MIN_VAL_COLOR = (255, 0, 0)

    EDGE_MIN_WIDHT = 0.1
    EDGE_MAX_WIDHT = 3

    dA_max = dw_max = w_max = output_max = input_max = -np.inf
    dA_min = dw_min = w_min = output_min = input_min = np.inf
    for i in range(len(nn_per_layer)):
        input = cache['Z'+str(i)]
        w = weights[i]
        output = cache['A'+str(i)]
        input_max = max(input.max(), input_max)
        input_min = min(input.min(), input_min)
        w_max = max(w.max(), w_max)
        w_min = min(w.min(), w_min)
        output_max = max(output.max(), output_max)
        output_min = min(output.min(), output_min)
        if i>0:
            dA = grads['dA_curr'+str(i)]
            dA_max = max(dA.max(), dA_max)
            dA_min = min(dA.min(), dA_min)
            dw = grads['dW'+str(i)]
            dw_max = max(dw.max(), dw_max)
            dw_min = min(dw.min(), dw_min)


    for i in range(0, len(nn_per_layer)):
        layer_act_func = model.layers[i].act_func
        labeljust = 'right'
        node_color = '#3498db'
        the_label = ""
        labelloc = 'b'
        edge_color = 'black'
        edge_width = 2

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
                if color_by_value:
                    if color_by_gradient and i>0:
                        node_color = value_to_color(dAcurr, dA_min, dA_max, NODE_MIN_VAL_COLOR,
                                                    NODE_MAX_VAL_COLOR)

                    else:
                        node_color = value_to_color(node_output, output_min, output_max, NODE_MIN_VAL_COLOR, NODE_MAX_VAL_COLOR)
                c.node(str(i) + ',' + str(j), tooltip=node_tooltip,
                       shape="circle", style="filled", fillcolor=node_color, fontcolor=node_color, color='black')
                if i != 0:
                    for h in range(last_layer_nodes):
                        w = weights[i][j][h]
                        node_before_output = cache['A' + str(i-1)][h][sample_id]
                        dw = grads['dW'+str(i)][j][h]
                        edge_tooltip = f"w: {w}\nbefore: {node_before_output}\nafter: {w*node_before_output}\n\nbackward pass:\ndw: {dw}"
                        if color_by_value:
                            if color_by_gradient:
                                edge_color = value_to_color(dw, dw_min, dw_max, NODE_MIN_VAL_COLOR, NODE_MAX_VAL_COLOR)
                            else:
                                edge_color = value_to_color(w, w_min, w_max, NODE_MIN_VAL_COLOR, NODE_MAX_VAL_COLOR)
                        edge_width = scale_value(np.abs(w), 0, max(np.abs(w_min), np.abs(w_max)), EDGE_MIN_WIDHT, EDGE_MAX_WIDHT)
                        g.edge(str(i - 1) + ',' + str(h), str(i) + ',' + str(j), tooltip=edge_tooltip, penwidth=str(edge_width), color=edge_color)
            last_layer_nodes = nn_per_layer[i]

    g.attr(arrowShape="none")
    g.edge_attr.update(arrowhead="none", color="#707070")
    g.node_attr.update(margin='0.1')
    if view == True:
        g.view()
    return g