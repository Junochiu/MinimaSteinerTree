import sys
import networkx as nx
import matplotlib.pyplot as plt
import os
import time


def find_shortest_pth_between_terminals(G, terminals):
    tic = time.time()
    '''
    create full connections between terminals first
    then update the path if a shorter version found
    set weight to maxima first
    and set path to NULL
    '''
    ter_G = nx.Graph()
    for u in terminals:
        for v in terminals:
            if u != v:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=1000000, path=[])
                    ter_G.add_edge(u, v, weight=1000000, path=[])
                else:
                    w = G[u][v]['weight']
                    G.remove_edge(u, v)
                    G.add_edge(u, v, weight=w, path=[u, v])
                    ter_G.add_edge(u, v, weight=w, path=[u, v])
    currentweight = ter_G.size(weight='weight')
    print("current weight {}".format(currentweight))
    newweight = 0
    # terminal for the terminal currently working on
    terminal_idx = 0
    ter_node = terminals[terminal_idx]

    '''
    stop if old total (cost - newcost)/cost < 0.05
    '''
    while (True):
        # for ter_node in terminals:
        # for each terminal node
        # visited = []
        print("current weight {}".format(currentweight))
        path = []
        stack = []
        numsofwalk = 0
        # initial the stack
        for node in G.neighbors(ter_node):
            stack.append(node)
        # visited.append(ter_node)
        path.append(ter_node)

        # if stack == 0 implies DFS finished
        while (len(stack) and numsofwalk < 5 * len(terminals)):
            # print("stack")
            # print(stack)
            if (len(path) == 0):
                path.append(ter_node)
            # print("current ter node {}".format(ter_node))
            cur_node = stack[-1]
            stack.pop()
            # print("current node {}".format(cur_node))
            path.append(cur_node)
            # print("line47 path {}".format(path))
            # if cur_node not in terminals:
            #    visited.append(cur_node)
            # print("visited {}".format(visited))
            lastingnode = []
            lastingroad = 0
            newcost = 0

            # prepare the unwalked path
            for node in G.neighbors(cur_node):
                # if node not in visited:
                if node not in path:
                    lastingroad = lastingroad + 1
                    lastingnode.append(node)

            if cur_node not in terminals and lastingroad == 0:
                # unused branch
                while len(path) != 0 and path[-1] != -1:
                    path.pop()
                if len(path) != 0 and path[-1] == -1:
                    path.pop()

            elif cur_node not in terminals and lastingroad > 0:
                # decision node
                # print("lasting node {}".format(lastingnode))
                for i in range(lastingroad):
                    path.append(-1)
                    stack.append(lastingnode[i])
                path.pop()

            else:
                # another terminal find
                numsofwalk = numsofwalk + 1
                terpath = []
                for i in path:
                    if i != -1:
                        terpath.append(i)
                print("path from {e1} to {e2} ".format(e1=ter_node, e2=cur_node))
                for i in range(len(terpath) - 1):
                    newcost = newcost + G[terpath[i]][terpath[i + 1]]['weight']
                # print("newcost={}".format(newcost))
                oldpath = G[ter_node][cur_node]['path']
                # print("oldpath {}".format(oldpath))
                '''
                if (len(oldpath)):
                    for i in range(len(oldpath) - 1):
                        oldcost = oldcost + G[oldpath[i]][oldpath[i + 1]]['weight']
                else:
                '''
                oldcost = G[ter_node][cur_node]['weight']
                if newcost < oldcost:
                    G.remove_edge(ter_node, cur_node)
                    G.add_edge(ter_node, cur_node, weight=newcost, path=terpath)
                    ter_G.remove_edge(ter_node, cur_node)
                    ter_G.add_edge(ter_node, cur_node, weight=newcost, path=terpath)
                    # print(G[ter_node][cur_node]['weight'])
                    # print("add edge from {ter} to {cur} with cost {cost}".format(ter=ter_node, cur=cur_node,
                    #                                                         cost=ter_G[ter_node][cur_node]['weight']))
                    # ter_G.add_edge(ter_node, cur_node, weight=newcost, path=terpath)

                    for edge in ter_G.edges:
                        u = edge[0]
                        v = edge[1]
                        cost = 0
                        pathlist = ter_G[u][v]['path']
                        if len(pathlist) > 0:
                            for i in range(len(pathlist) - 1):
                                cost = cost + G[pathlist[i]][pathlist[i + 1]]['weight']
                            ter_G[u][v]['weight'] = cost

                # renew the walking part
                while len(path) != 0 and path[-1] != -1:
                    path.pop()
                if len(path) != 0 and path[-1] == -1:
                    path.pop()
        if len(stack) == 0:
            print("beacuse of stack empty")
        else:
            print("because run too many times")
        terminal_idx = (terminal_idx + 1) % len(terminals)
        ter_node = terminals[terminal_idx]
        newweight = ter_G.size(weight='weight')
        # print("newweight {}".format(newweight))
        if abs((currentweight - newweight)) < 0.01 * currentweight:
            break
        currentweight = newweight
    toc = time.time()
    print("finding shortestpath {}".format(toc - tic))
    # drawing the graph
    '''
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=80)
    nx.draw_networkx_edges(G, pos, width=1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    nx.draw_networkx_nodes(ter_G, pos, node_size=80)
    nx.draw_networkx_edges(ter_G, pos, width=1)
    nx.draw_networkx_labels(ter_G, pos, font_size=10, font_family='sans-serif')
    labels = nx.get_edge_attributes(ter_G, 'weight')
    nx.draw_networkx_edge_labels(ter_G, pos, edge_labels=labels)
    plt.show()
    '''
    return ter_G


def min_spanning_tree(ter_G):
    tic = time.time()
    tmp_adj_matrix = []
    tmp_adj_matrix_node = []
    adj_matrix = nx.to_numpy_array(ter_G)
    spanningtree = nx.Graph()
    # print(ter_G.nodes)
    nodelist = []
    for node in ter_G.nodes:
        nodelist.append(node)
    # print(nodelist[0])
    tmp_adj_matrix.append(adj_matrix[0])
    tmp_adj_matrix_node.append(nodelist[0])

    for k in range(len(nodelist) - 1):
        min = 100000000
        for i in range(len(tmp_adj_matrix)):
            for j in range(len(tmp_adj_matrix[i])):
                if tmp_adj_matrix[i][j] != 0 and nodelist[j] not in tmp_adj_matrix_node:
                    if tmp_adj_matrix[i][j] < min:
                        mini = i
                        minj = j
                        min = tmp_adj_matrix[i][j]
        tmp_adj_matrix.append(adj_matrix[minj])
        tmp_adj_matrix_node.append(nodelist[minj])
        u = tmp_adj_matrix_node[mini]
        v = nodelist[minj]
        spanningtree.add_edge(u, v, weight=ter_G[u][v]['weight'], path=ter_G[u][v]['path'])
    '''
    pos = nx.spring_layout(spanningtree)
    nx.draw_networkx_nodes(spanningtree, pos, node_size=80)
    nx.draw_networkx_edges(spanningtree, pos, width=1)
    nx.draw_networkx_labels(spanningtree, pos, font_size=10, font_family='sans-serif')
    # labels = nx.get_edge_attributes(ter_G, 'weight')
    # nx.draw_networkx_edge_labels(ter_G, pos, edge_labels=labels)
    plt.show()
    '''
    toc = time.time()
    print("MST1 {}".format(toc - tic))
    return spanningtree


def mergeSort(edgelist):
    if len(edgelist) > 1:
        mid = len(edgelist) // 2  # Finding the mid of the array
        L = edgelist[:mid]  # Dividing the array elements
        R = edgelist[mid:]  # into 2 halves

        mergeSort(L)  # Sorting the first half
        mergeSort(R)  # Sorting the second half

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i][2] < R[j][2]:
                edgelist[k] = L[i]
                i += 1
            else:
                edgelist[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            edgelist[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            edgelist[k] = R[j]
            j += 1
            k += 1


def min_spanning_tree_1(ter_G):
    tic = time.time()
    edgelist = []
    for edge in ter_G.edges:
        nodedata = []
        nodedata.append(edge[0])
        nodedata.append(edge[1])
        nodedata.append(ter_G[edge[0]][edge[1]]['weight'])
        nodedata.append(ter_G[edge[0]][edge[1]]['path'])
        edgelist.append(nodedata)
    mergeSort(edgelist)
    # print(edgelist)
    spanning = nx.Graph()
    treeid = 0
    for edge in edgelist:
        u = edge[0]
        v = edge[1]
        has_u = spanning.has_node(edge[0])
        has_v = spanning.has_node(edge[1])
        if has_u and has_v:
            id_v = spanning.nodes[v]['id']
            id_u = spanning.nodes[u]['id']
            if spanning.nodes[u]['id'] != spanning.nodes[v]['id']:
                spanning.add_edge(u, v, weight=edge[2], path=edge[3])
                spanning.nodes[v]['id'] = id_u
                for node in spanning.nodes:
                    if spanning.nodes[node]['id'] == id_v:
                        spanning.nodes[node]['id'] = spanning.nodes[u]['id']
        elif has_u:
            spanning.add_node(v, id=spanning.nodes[u]['id'])
            spanning.add_edge(u, v, weight=edge[2], path=edge[3])
        elif has_v:
            spanning.add_node(u, id=spanning.nodes[v]['id'])
            spanning.add_edge(u, v, weight=edge[2], path=edge[3])
        else:
            spanning.add_node(u, id=treeid)
            spanning.add_node(v, id=treeid)
            spanning.add_edge(u, v, weight=edge[2], path=edge[3])
            treeid = treeid + 1
    '''
    pos = nx.spring_layout(spanning)
    nx.draw_networkx_nodes(spanning, pos, node_size=80)
    nx.draw_networkx_edges(spanning, pos, width=1)
    nx.draw_networkx_labels(spanning, pos, font_size=10, font_family='sans-serif')
    plt.show()
    '''
    toc = time.time()
    print("spanning1 {}".format(toc - tic))
    return spanning


def min_spanning_tree_3(ter_G):
    tic = time.time()
    edgelist = []
    for edge in ter_G.edges:
        nodedata = []
        nodedata.append(edge[0])
        nodedata.append(edge[1])
        nodedata.append(ter_G[edge[0]][edge[1]]['weight'])
        edgelist.append(nodedata)
    mergeSort(edgelist)
    # print(edgelist)
    spanning = nx.Graph()
    treeid = 0
    for edge in edgelist:
        u = edge[0]
        v = edge[1]
        has_u = spanning.has_node(edge[0])
        has_v = spanning.has_node(edge[1])
        if has_u and has_v:
            id_v = spanning.nodes[v]['id']
            id_u = spanning.nodes[u]['id']
            if spanning.nodes[u]['id'] != spanning.nodes[v]['id']:
                spanning.add_edge(u, v, weight=edge[2])
                spanning.nodes[v]['id'] = id_u
                for node in spanning.nodes:
                    if spanning.nodes[node]['id'] == id_v:
                        spanning.nodes[node]['id'] = spanning.nodes[u]['id']
        elif has_u:
            spanning.add_node(v, id=spanning.nodes[u]['id'])
            spanning.add_edge(u, v, weight=edge[2])
        elif has_v:
            spanning.add_node(u, id=spanning.nodes[v]['id'])
            spanning.add_edge(u, v, weight=edge[2])
        else:
            spanning.add_node(u, id=treeid)
            spanning.add_node(v, id=treeid)
            spanning.add_edge(u, v, weight=edge[2])
            treeid = treeid + 1
    '''
    pos = nx.spring_layout(spanning)
    nx.draw_networkx_nodes(spanning, pos, node_size=80)
    nx.draw_networkx_edges(spanning, pos, width=1)
    nx.draw_networkx_labels(spanning, pos, font_size=10, font_family='sans-serif')
    plt.show()
    '''
    toc = time.time()
    print("spanning {}".format(toc - tic))
    return spanning


def to_graph_with_steiner_points(tree, G):
    tic = time.time()
    stein_G = nx.Graph()
    for u in tree.nodes:
        # print("node {}".format(u))
        for v in tree.adj[u]:
            nodelist = tree[u][v]['path']
            # print("path to {v} = {path}".format(v=v, path=nodelist))
            for i in range(len(nodelist) - 1):
                x = nodelist[i]
                y = nodelist[i + 1]
                stein_G.add_edge(x, y, weight=G[x][y]['weight'])
    '''
    pos = nx.spring_layout(stein_G)
    nx.draw_networkx_nodes(stein_G, pos, node_size=80)
    nx.draw_networkx_edges(stein_G, pos, width=1)
    nx.draw_networkx_labels(stein_G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    '''
    toc = time.time()
    print("to steiner {}".format(toc - tic))
    return stein_G


def main():
    tStart = time.time()
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
    stein_G = to_graph_with_steiner_points(tree, G)
    # stein_G = min_spanning_tree_2(stein_G)
    stein_G = min_spanning_tree_3(stein_G)

    end = False
    while end != True:
        end = True
        for degree in list(stein_G.degree()):
            if degree[1] == 1 and degree[0] not in terminals:
                stein_G.remove_node(degree[0])
                end = False
    # print(ter_G.nodes)
    # print(nx.to_numpy_matrix(ter_G))
    # print(ter_G[37][48]['path'])
    # print(ter_G[48][37]['path'])
    s = sys.argv[1].split("/")
    filename = s[2] + ".outputs"
    if not os.path.exists('output'):
        os.makedirs('output')
    outputfile = open(os.path.join("output", filename), 'a+')

    for edge in stein_G.edges:
        u = edge[0]
        v = edge[1]
        out = str(u) + " " + str(v) + "\n"
        outputfile.write(out)
    outputfile.close()

    tEnd = time.time()
    print("time {}".format(tEnd - tStart))


if __name__ == '__main__':
    main()
