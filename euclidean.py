import networkx as nx
#import matplotlib.pyplot as plt
import os
import sys
import math


def to_connected_graph(graph):
    for u in graph.nodes:
        for v in graph.nodes:
            if u != v:
                w = cal_distance(graph, u, v)
                graph.add_edge(u, v, weight=w)
    return graph


def cal_distance(graph, id1, id2):
    pos1 = graph.nodes[id1]['pos']
    pos2 = graph.nodes[id2]['pos']
    distance = (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]) + (
            pos1[2] - pos2[2]) * (pos1[2] - pos2[2])
    distance = math.sqrt(distance)
    return distance


def precalculate():
    unitx = [1, 0]
    unity = [0, 1]
    rotatex = []
    rotatey = []
    rotatex.append(math.cos(math.radians(-60)) * unitx[0] - math.sin(math.radians(-60)) * unitx[1])
    rotatex.append(math.sin(math.radians(-60)) * unitx[0] + math.cos(math.radians(-60)) * unitx[1])
    rotatey.append(math.cos(math.radians(60)) * unity[0] - math.sin(math.radians(60)) * unity[1])
    rotatey.append(math.sin(math.radians(60)) * unity[0] + math.cos(math.radians(60)) * unity[1])
    a = (unity[1] - rotatex[1]) / (unity[0] - rotatex[0])
    c = (unitx[1] - rotatey[1]) / (unitx[0] - rotatey[0])
    b = (-1 * a * unity[0]) + unity[1]
    d = (-1 * b * unitx[0]) + unitx[1]
    timesy = ((((-1 * d * a) / c)) + b) / (1 - (a / c))
    timesx = (timesy - d) / c
    # find cross point between line(x,rotatey) and line(y,rotatex)
    return timesx, timesy


def optimize(posx, posy, posz):
    #precalculate()
    vecyx = [posx[i] - posy[i] for i in range(len(posx))]
    vecyz = [posz[i] - posy[i] for i in range(len(posy))]
    precalculateconstant = [0.21132486540518716, 0.21132486540518713]
    fermatpoint = []
    for i in range(3):
        fermatpoint.append(precalculateconstant[0] * vecyx[i] + precalculateconstant[1] * vecyz[i] + posy[i])

    return fermatpoint


def find_steinerpoint(graph, totalnodecount):
    currnetweight = graph.size(weight='weight')
    newweight = 0
    stein_point = []
    while True:
        currentweight = newweight
        for degree in list(graph.degree):
            find = False
            if degree[1] >= 2:
                y = degree[0]
                minangle = math.radians(120)
                for x in graph.neighbors(degree[0]):
                    for z in graph.neighbors(degree[0]):
                        if x != z:
                            posx = graph.nodes[x]['pos']
                            posy = graph.nodes[y]['pos']
                            posz = graph.nodes[z]['pos']
                            dir1 = []
                            dir2 = []
                            for i in range(3):
                                dir1.append(posz[i] - posx[i])
                                dir2.append(posy[i] - posx[i])
                            numerator = 0
                            length1 = 0
                            length2 = 0
                            for i in range(3):
                                numerator = numerator + dir1[i] * dir2[i]
                                length1 = length1 + dir1[i] * dir1[i]
                                length2 = length2 + dir2[i] * dir2[i]
                            denominator = math.sqrt(length1) + math.sqrt(length2)
                            if denominator!=0 and abs(numerator / denominator)<=1:
                                if math.acos(numerator / denominator) < minangle:
                                    minangle = math.acos(numerator / denominator)
                                    minangleX = x
                                    minangleZ = z
                                    find = True
                # find optimize point
                x = minangleX
                z = minangleZ
                if find == True:
                    optimized_pos = optimize(posx, posy, posz)
                    yneighbors = list(graph.neighbors(y))
                    graph.remove_edge(y, x)
                    graph.remove_edge(y, z)

                    if y in stein_point:
                        steinerpointname = y
                        graph.remove_node(y)
                        graph.add_node(steinerpointname, pos=optimized_pos)
                        for node in yneighbors:
                            graph.add_edge(steinerpointname, node, weight=cal_distance(graph, node, steinerpointname))
                    else:
                        steinerpointname = totalnodecount
                        totalnodecount = totalnodecount + 1
                        graph.add_node(steinerpointname, pos=optimized_pos)
                        graph.add_edge(steinerpointname, x, weight=cal_distance(graph, x, steinerpointname))
                        graph.add_edge(steinerpointname, y, weight=cal_distance(graph, y, steinerpointname))
                        graph.add_edge(steinerpointname, z, weight=cal_distance(graph, z, steinerpointname))
                        stein_point.append(steinerpointname)
                    newweight = graph.size(weight='weight')
        if abs(currentweight - newweight) < 0.05:
            break
    return graph, stein_point


def min_spanning_tree(graph):
    tmp_adj_matrix = []
    tmp_adj_matrix_node = []
    adj_matrix = nx.to_numpy_array(graph)
    spanningtree = nx.Graph()
    nodelist = []
    for node in graph.nodes:
        nodelist.append(node)
    tmp_adj_matrix.append(adj_matrix[0])
    tmp_adj_matrix_node.append(nodelist[0])
    for k in range(len(nodelist) - 1):
        min = 100000000
        for i in range(len(tmp_adj_matrix)):
            for j in range(len(tmp_adj_matrix[i])):
                if tmp_adj_matrix[i][j] > 0 and nodelist[j] not in tmp_adj_matrix_node:
                    if tmp_adj_matrix[i][j] < min:
                        mini = i
                        minj = j
                        min = tmp_adj_matrix[i][j]
        tmp_adj_matrix.append(adj_matrix[minj])
        tmp_adj_matrix_node.append(nodelist[minj])
        u = tmp_adj_matrix_node[mini]
        v = nodelist[minj]
        spanningtree.add_node(u, pos=graph.nodes[u]['pos'])
        spanningtree.add_node(v, pos=graph.nodes[v]['pos'])
        spanningtree.add_edge(u, v, weight=graph[u][v]['weight'])
    '''
    pos = nx.spring_layout(spanningtree)
    nx.draw_networkx_nodes(spanningtree, pos, node_size=80)
    nx.draw_networkx_edges(spanningtree, pos, width=1)
    nx.draw_networkx_labels(spanningtree, pos, font_size=10, font_family='sans-serif')
    labels = nx.get_edge_attributes(spanningtree, 'weight')
    nx.draw_networkx_edge_labels(spanningtree, pos, edge_labels=labels)
    plt.show()
    '''
    return spanningtree


def main():
    totalnodecount = 1
    steinertree_file = open(sys.argv[1], "r")
    G = nx.Graph()
    firstline = True
    i = 0
    # initialize the graph
    for line in steinertree_file:
        pos = []
        if firstline:
            terminalnum = int(line.split()[0])
            firstline = False
        else:
            pos.append(float(line.split()[0]))
            pos.append(float(line.split()[1]))
            pos.append(float(line.split()[2]))
            nodename = 'T' + str(i)
            i = i + 1
            G.add_node(totalnodecount, pos=pos)
            totalnodecount = totalnodecount + 1
    # create connected graph
    G = to_connected_graph(G)
    labels = nx.get_node_attributes(G, 'pos')
    G = min_spanning_tree(G)
    G, stein_list = find_steinerpoint(G, totalnodecount)

    s = sys.argv[1].split("/")
    filename = s[2] + ".outputs"
    if not os.path.exists('output'):
        os.makedirs('output')
    outputfile = open(os.path.join("output", filename), 'a+')
    for node in stein_list:
        stein_pos = G.nodes[node]['pos']
        outputfile.write(str(stein_pos[0]) + " " + str(stein_pos[1]) + " " + str(stein_pos[2]) + "\n")
    firstLine = True
    for edge in G.edges:
        if firstLine == False:
            outputfile.write(", ")
        outputfile.write(str(edge[0]) + "-" + str(edge[1]))
        firstLine = False
    outputfile.close()
    '''
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=80)
    nx.draw_networkx_edges(G, pos, width=1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    '''

if __name__ == '__main__':
    main()
