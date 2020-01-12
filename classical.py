import sys
import networkx as nx
import matplotlib.pyplot as plt


def find_shortest_pth_between_terminals(G, terminals):
    terminal_num = max(terminals)
    weightmatrix = [[100000 for j in range(terminal_num)] for i in range(terminal_num)]
    # print(weightmatrix)
    ter_G = nx.Graph()

    for node in terminals:
        walkingstack = []
        decisionpoint = []
        currentnode = node
        walkingstack.append(currentnode)

        finish = False
        while not finish:
            lastingroad = 0
            print("currentnode")
            print(currentnode)
            for adjnode in list(G.adj[currentnode]):
                if adjnode not in walkingstack:
                    lastingroad = lastingroad + 1
            if (currentnode not in terminals) and (lastingroad == 0):
                if currentnode == node:
                    finish = True
                else:
                    while walkingstack[-1] != -1:
                        walkingstack.pop()
                    walkingstack.pop()
                    currentnode = decisionpoint[-1]
                    decisionpoint.pop()
            elif (currentnode != node) and (currentnode in terminals):
                tmppath = []
                cost = 0
                walkingstack.append(currentnode)
                print("walkingstack")
                print(walkingstack)
                print("decision")
                print(decisionpoint)
                count = 0
                for j in range(len(walkingstack)):
                    if walkingstack[j] != -1:
                        tmppath.append(walkingstack[j])
                # print(tmppath)
                # print(list(G.nodes))
                for j in range(len(tmppath) - 1):
                    print(tmppath[j])
                    print(tmppath[j + 1])
                    print(G[tmppath[j]][tmppath[j + 1]]['weight'])
                    print(float(G[tmppath[j]][tmppath[j + 1]]['weight']))
                    cost = cost + float(G[tmppath[j]][tmppath[j + 1]]['weight'])
                    print(cost)
                if cost < weightmatrix[int(node) - 1][int(currentnode) - 1]:
                    weightmatrix[int(node) - 1][int(currentnode) - 1] = cost
                    weightmatrix[int(currentnode) - 1][int(node) - 1] = cost
                    ter_G.add_edge(node, currentnode, weight=cost, path=tmppath)

                if len(decisionpoint) == 0 or lastingroad == 0:
                    finish = True
                else:
                    i = len(walkingstack) - 1
                    while walkingstack[i] != -1:
                        walkingstack.pop()
                        i = i - 1
                    walkingstack.pop()
                    currentnode = decisionpoint[len(decisionpoint) - 1]
                    decisionpoint.pop()

            else:
                if G.degree[currentnode] > 0:
                    if currentnode != node and G.degree[currentnode] != 1:
                        walkingstack.append(currentnode)
                    for adjnode in list(G.adj[currentnode]):
                        if adjnode not in walkingstack:
                            decisionpoint.append(adjnode)
                            walkingstack.append(-1)
                    # move to the next node
                    currentnode = decisionpoint[-1]
                    decisionpoint.pop()
                    walkingstack.pop()

    # drawing the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=80)
    nx.draw_networkx_edges(G, pos, width=1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    nx.draw_networkx_nodes(ter_G, pos, node_size=80)
    nx.draw_networkx_edges(ter_G, pos, width=1)
    nx.draw_networkx_labels(ter_G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    return ter_G


def min_spanning_tree(ter_G):
    tmp_adj_matrix = []
    tmp_adj_matrix_node = []
    adj_matrix = nx.to_numpy_array(ter_G)
    spanningtree = nx.Graph()
    print(ter_G.nodes)
    nodelist = []
    for node in ter_G.nodes:
        nodelist.append(node)
    print(nodelist[0])
    tmp_adj_matrix.append(adj_matrix[0])
    tmp_adj_matrix_node.append(nodelist[0])

    for k in range(len(nodelist)):
        for i in range(len(tmp_adj_matrix)):
            max = 10000
            for j in range(len(tmp_adj_matrix[i])):
                if tmp_adj_matrix[i][j] != 0 and nodelist[j] not in tmp_adj_matrix_node:
                    if tmp_adj_matrix[i][j] < max:
                        mini = i
                        minj = j
                        max = adj_matrix[i][j]
        tmp_adj_matrix.append(adj_matrix[minj])
        tmp_adj_matrix_node.append(nodelist[minj])
        u = tmp_adj_matrix_node[mini]
        v = nodelist[minj]
        spanningtree.add_edge(u, v, weight=ter_G[u][v]['weight'], path=ter_G[u][v]['path'])

    pos = nx.spring_layout(spanningtree)
    nx.draw_networkx_nodes(spanningtree, pos, node_size=80)
    nx.draw_networkx_edges(spanningtree, pos, width=1)
    nx.draw_networkx_labels(spanningtree, pos, font_size=10, font_family='sans-serif')
    plt.show()

    return spanningtree


def to_graph_with_steiner_points(tree):
    stein_G = nx.Graph()
    for u in tree.nodes:
        for v in tree.adj[u]:
            nodelist = tree[u][v]['path']
            print(nodelist)
            for i in range(len(nodelist) - 1):
                print(i)
                print(nodelist[i])
                print(nodelist[i + 1])
                x = nodelist[i]
                y = nodelist[i + 1]
                stein_G.add_edge(x, y)

    pos = nx.spring_layout(stein_G)
    nx.draw_networkx_nodes(stein_G, pos, node_size=80)
    nx.draw_networkx_edges(stein_G, pos, width=1)
    nx.draw_networkx_labels(stein_G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    return stein_G


def deal_with_cycle(stein_G, G):
    maxweight = 0

    try:
        if len(nx.find_cycle(stein_G)) > 0:
            havecycle = True
        else:
            havecycle = False
        while havecycle:
            for (u, v) in nx.find_cycle(stein_G):
                if G[u][v]['weight'] > maxweight:
                    maxweight = G[u][v]['weight']
                    tu = u
                    tv = v
            stein_G.remove_edge(tu, tv)

    except:
        havecycle = False

    pos = nx.spring_layout(stein_G)
    nx.draw_networkx_nodes(stein_G, pos, node_size=80)
    nx.draw_networkx_edges(stein_G, pos, width=1)
    nx.draw_networkx_labels(stein_G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    return stein_G


def main():
    steinertree_file = open(sys.argv[1], "r")
    terminals_file = open(sys.argv[2], "r")
    G = nx.Graph()

    for line in steinertree_file:
        node1 = int(line.split()[0])
        node2 = int(line.split()[1])
        w = float(line.split()[2])
        # print(node1)
        G.add_edge(node1, node2, weight=w)
        # print(G.nodes)

    terminals = []
    for line in terminals_file:
        terminal = line.split()[0]
        terminals.append(int(terminal))
        # print(terminals)
    ter_G = find_shortest_pth_between_terminals(G, terminals)
    tree = min_spanning_tree(ter_G)
    stein_G = to_graph_with_steiner_points(tree)
    stein_G = deal_with_cycle(stein_G, G)
    print(ter_G.nodes)
    print(nx.to_numpy_matrix(ter_G))
    print(ter_G[37][48]['path'])
    print(ter_G[48][37]['path'])


if __name__ == '__main__':
    main()
