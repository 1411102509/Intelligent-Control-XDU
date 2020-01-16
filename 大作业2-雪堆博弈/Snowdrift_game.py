import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt

node_size = 20
edge_sparse = 0.5 # 0-1
node_colors = []

# 随机生成一个网络
G = nx.Graph()
for i in range(node_size):
    # 初始化网络，随机选策略
    if random.random() > 0.5:
        choice = 0  # 选择策略为C
        node_colors.append('r')
    else:
        choice = 1  # 选择策略为D
        node_colors.append('c')
    
    # 添加节点
    G.add_node(i, name = i, strategy = choice, porfit = 0)

    # 随机添加边
    if i != 0:
        G.add_edge(i,i-1)
        if random.random() < edge_sparse:
            G.add_edge(i, random.randint(0,len(G.nodes)-1))

# 统计网络的最大度
max_degree = 0
for i in range(node_size):
    if G.degree(i) > max_degree:
        max_degree = G.degree(i)

# 显示初始网络
plt.figure("初始网络")
nx.draw_networkx(G, node_color = node_colors)
plt.show(block = False)

# 雪堆博弈收益矩阵
# r = 0.5
r = 1/max_degree * (0.9)
porfit_mat = [[(1,1),(1-r,1+r)],[(1+r,1-r),(0,0)]]

# 不断博弈，找到网络博弈的纳什均衡
itera = 0
change = 1
while change:
    change = 0
    itera += 1
    if itera > 100:
        print("该网络没有纳什均衡")
        exit()

    for n, nbrs in G.adjacency():   # n:遍历的第n个节点，nbrs：所有邻接节点
        # print("*****************",n,"********************")
        porfit_0 = 0    # 选择0（Cooperator）策略的收益
        porfit_1 = 0    # 选择1（Defector）策略的收益
        for nbr,_ in nbrs.items():
            # print("n_strategy:",G.nodes[n]['strategy'],"    nbr_",nbr,"strategy:",G.nodes[nbr]['strategy'])
            porfit_0 += porfit_mat[0][G.nodes[nbr]['strategy']][0]
        for nbr,_ in nbrs.items():
            porfit_1 += porfit_mat[1][G.nodes[nbr]['strategy']][0]
        
        if porfit_0 >= porfit_1:
            if G.nodes[n]['strategy'] == 1:
                change = 1
            G.nodes[n]['strategy'] = 0
            node_colors[n] = 'r'
        else:
            if G.nodes[n]['strategy'] == 0:
                change = 1
            G.nodes[n]['strategy'] = 1
            node_colors[n] = 'c'

# for i in range(node_size):
#     print(G.nodes(data = True)[i])
# 输出图像展示
print("网络的最大度：",max_degree)
plt.figure("网络博弈的纳什均衡")
nx.draw_networkx(G, node_color = node_colors)
plt.show()
