'''
This program will create a Euclidean Steiner Tree on test graph data.
Write: Wei-Chian Liang
Date:2020.01.12
'''
import sys  # for input from cmd line
import os # for get the file route
import networkx as nx   # for build graph
from networkx.algorithms import approximation as ax
import numpy as np  
import matplotlib.pyplot as plt # for polt graph
import timeit   # for record program time
import math

steadyThread = 0.01
start = timeit.default_timer()
#===============Functions==================
def plotGraph(graph):
    #pos=nx.spring_layout(graph)
    #pos=nx.shell_layout(graph)
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph,pos,with_labels=True, node_size=400, alpha=0.5 )
    labels = nx.get_edge_attributes(graph,'weight')
    #nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)
    plt.show()

def getDistance(P1, P2):
    return math.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2 + (P1[2]-P2[2])**2)

def GetCrossAngle(v1, v2):
    cos_value = (float(v1.dot(v2)) / (np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2))))
    return np.arccos(cos_value) * (180/np.pi)

def rotate(angle, point):
    valuex = point[0]
    valuey = point[1]
    rotatex = math.cos(angle)*valuex -math.sin(angle)*valuey
    rotatey = math.cos(angle)*valuey + math.sin(angle)*valuex
    return [rotatex, rotatey]

def cross_point(line1,line2):
    x1=line1[0]
    y1=line1[1]
    x2=line1[2]
    y2=line1[3]
    
    x3=line2[0]
    y3=line2[1]
    x4=line2[2]
    y4=line2[3]
    
    k1=(y2-y1)*1.0/(x2-x1)
    b1=y1*1.0-x1*k1*1.0
    if (x4-x3)==0:
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0
    return [x,y]

def getFermatPoint (P1, P2, P3):
    # Take P2 as original point
    vec21 = P1 - P2
    print("infermat point")
    print(vec21)
    vec23 = P3 - P2
    print(vec23)
    '''
    newP1 = [1, 0]
    newP3 = [0, 1]
    # Get new newP4 on new coordinate by rotate newP1 clockwise
    newP4 = rotate (math.pi*(-60/180), newP1)
    # Get new newP5 on new coordinate by rotate newP3 anti-clockwise
    newP5 = rotate (math.pi*(60/180), newP3)
    # Get the cross point on new corrdinate
    print ("newP4=",newP4,",newP5=",newP5)
    line1 = [newP3[0], newP3[1], newP4[0], newP4[1]]
    line2 = [newP5[0], newP5[1], newP1[0], newP1[1]]
    newCrossP = cross_point(line1, line2)
    print ("croosP=",newCrossP)
    '''
    newCrossP = [0.21132486540518716, 0.21132486540518713] # the result from above calculate is a constant
    # Mapping newCrossP back to original coordinate
    fermatPoint = newCrossP[0]*vec21 + newCrossP[1]*vec23 + P2
    return fermatPoint

# Get the first in the list
def takeFirst(elem):
    return elem[0]

#===============Start===================
def run():
    # Read 3-D coordinate from file
    firstLine = True
    totalNode = 0
    coordinate = []
    with open(sys.argv[1]) as file:
        for line in file:
            line = line.strip().split()
            if firstLine:
                totalNode = int(line[0])
                firstLine = False
                continue
            coordinate.append(line)
    coordinate = np.array (coordinate, dtype=np.float64)

    # Create a graph by distance
    G = nx.Graph()
    maxNum = 0
    for a in range(totalNode):
        for b in range(a+1,totalNode):
            distance = getDistance(coordinate[a], coordinate[b])
            G.add_node (a+1, pos = (coordinate[a][0], coordinate[a][1]), STnode = False)
            G.add_node (b+1, pos = (coordinate[b][0], coordinate[b][1]), STnode = False)
            G.add_edge (a+1, b+1, weight = distance)
            if distance > maxNum:
                maxNum = distance

    # Build a minimum spanning tree on whole nodes
    st_tree = nx.Graph()
    p = nx.get_node_attributes(G, 'pos')
    st_tree.add_node(1, pos = p[1], STnode = False)
    while st_tree.number_of_nodes() != G.number_of_nodes():
        minOldNodeLoc = None
        minNewNodeLoc = None
        minWeightLoc = maxNum
        for node in list(st_tree.nodes()):
            for neighbor in G.neighbors(node):
                if st_tree.has_node(neighbor) == False \
                    and G[node][neighbor]['weight'] < minWeightLoc:
                        minWeightLoc = G[node][neighbor]['weight']
                        minOldNodeLoc = node
                        minNewNodeLoc = neighbor
        if minNewNodeLoc != None:
            st_tree.add_node(minNewNodeLoc, pos = p[minNewNodeLoc], STnode = False)
            st_tree.add_edge(minOldNodeLoc, minNewNodeLoc, weight = minWeightLoc)

    # Find Steiner node
    minTotalWeight = st_tree.size(weight='weight')
    newTotalWeight = minTotalWeight
    #print("Original total wight=", minTotalWeight)
    #plotGraph (st_tree)
    while True:
        minTotalWeight = newTotalWeight
        for degree in list (st_tree.degree()):
            minNodeA = minNodeB = minNodeM = -1
            minAngle = 120
            if degree[1] >= 2 :
                vec = []
                # Find all vector from node(degree[0]) to its neighbor
                for neighbor in st_tree.neighbors(degree[0]):
                    #print (coordinate[neighbor-1], ", ", coordinate[degree[0]-1])
                    vec.append(coordinate[neighbor-1] - coordinate[degree[0]-1])
                # Find the vector pair with the smallest angle
                a = 0
                for vec1 in vec[:int(len(vec)/2)]:
                    b = int(len(vec)/2)
                    for vec2 in vec[int(len(vec)/2):]:
                        if GetCrossAngle(vec1, vec2) < minAngle:
                            minAngle = GetCrossAngle(vec1, vec2)
                            nei = list(st_tree.neighbors(degree[0]))
                            minNodeA = nei[a]
                            minNodeB = nei[b]   
                            minNodeM = degree[0]
                        #print ("vec1=", vec1, ", vec2=", vec2, ", minAngle=", minAngle)
                        b += 1
                    a += 1
                # Find Steiner node(Fermat point) and renew the graph
                STnodeFlag = nx.get_node_attributes(st_tree, 'STnode')
                if minAngle < 120:
                    #print("minNodeA= ", minNodeA, ", minNodeM=", minNodeM, ", minNodeB= ", minNodeB)
                    FMcoordinate = getFermatPoint(coordinate[minNodeA-1], coordinate[minNodeM-1], coordinate[minNodeB-1])
                    #print ("Fermat point=", FMcoordinate)
                    Mneighbor = list(st_tree.neighbors(minNodeM))
                    # Remove originial edges
                    st_tree.remove_edge(minNodeM, minNodeA)
                    st_tree.remove_edge(minNodeM, minNodeB)
                    # Add new ST node

                    STnodeNum = -1
                    if STnodeFlag[minNodeM] == False:
                        # The minNodeM is not a ST node -> add ST node
                        totalNode += 1
                        STnodeNum = totalNode
                        coordinate = coordinate.tolist()
                        coordinate.append(FMcoordinate)
                        coordinate = np.array (coordinate)
                    else:
                        # The minNodeM is already a ST node -> renew ST node coordinate, and remove it
                        coordinate[minNodeM-1] = FMcoordinate
                        st_tree.remove_node(minNodeM)
                        STnodeNum = minNodeM
                    st_tree.add_node(STnodeNum, pos = (FMcoordinate[0], FMcoordinate[1]), STnode = True)
                    # Reconnect neighbors
                    if STnodeNum != minNodeM:
                        # Original: not a ST node
                        st_tree.add_edge(STnodeNum, minNodeM, weight = getDistance(FMcoordinate, coordinate[minNodeM-1]))
                        st_tree.add_edge(STnodeNum, minNodeA, weight = getDistance(FMcoordinate, coordinate[minNodeA-1]))
                        st_tree.add_edge(STnodeNum, minNodeB, weight = getDistance(FMcoordinate, coordinate[minNodeB-1]))
                    else:
                        # Original: ST node
                        for nei in Mneighbor:
                            st_tree.add_edge(STnodeNum, nei, weight = getDistance(FMcoordinate, coordinate[nei-1]))
                    newTotalWeight = st_tree.size(weight='weight')
                    #plotGraph (st_tree)
                    #print("total wight=", newTotalWeight)
            #print("~~")
        #print ("minTotalWeight=", minTotalWeight, "newTotalWeight=", newTotalWeight)
        #print ("=============")
        if abs(minTotalWeight - newTotalWeight) < steadyThread:
            break

    # Write result into file
    outFileName = sys.argv[1].split('/')
    outFileName = outFileName[-1]
    outFile = open(os.path.join("output", outFileName+".outputs"), "w")
    # Write the ST node
    isSTnode = nx.get_node_attributes(st_tree, 'STnode')
    for i in range(1, totalNode+1):
        if isSTnode[i] == True:
            outFile.write(str(coordinate[i-1][0])+" "+str(coordinate[i-1][1])+" "+str(coordinate[i-1][2])+"\n")
    # Sort the edges
    edges = list(st_tree.edges())
    outEdges =[]
    for out in edges:
        if out[0] < out[1]:
            outEdges.append([out[0], out[1]])
        else:
            outEdges.append([out[1], out[0]])
    outEdges.sort(key=takeFirst)
    # Write the edges
    firstLine = True
    for out in outEdges:
        if firstLine == False:
            outFile.write(", ")
        outFile.write(str(out[0])+"-"+str(out[1]))
        firstLine = False
    outFile.close()

    '''    
    # Analysis
    print("total wight=", st_tree.size(weight='weight'))
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    '''
    #plotGraph (st_tree)
    

if __name__ == "__main__":
    run()