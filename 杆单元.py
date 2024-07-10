import numpy as np

class Truss2D:
    def __init__(self):
        self.nodes = {}  # 存储节点的坐标
        self.elements = []  # 存储杆的属性和连接节点
        self.loads = {}  # 存储节点上的外力，包括大小和方向
        self.node_count = 0  # 节点数量
        self.K = None  # 全局刚度矩阵
        self.displacements = None  # 存储节点位移结果
        self.boundary = []


    def add_node(self, node_id, x, y):
        """ 添加节点，并指定其坐标 """
        self.nodes[node_id] = (x, y)
        self.node_count = max(self.node_count, node_id)

    def add_element(self, node1, node2, E, A):
        """ 添加杆元，定义其连接节点和材料属性 """
        self.elements.append({'node1': node1, 'node2': node2, 'E': E, 'A': A})

    def add_load(self, node, fx, fy):
        """ 在指定节点添加外力，包括x和y方向 """
        self.loads[node] = (fx, fy)

    def assemble_stiffness(self):
        """ 计算全局刚度矩阵 """
        n_dof = 2 * self.node_count  # 每个节点两个自由度
        self.K = np.zeros((n_dof, n_dof))
        for element in self.elements:
            node1 = element['node1']
            node2 = element['node2']
            x1, y1 = self.nodes[node1]
            x2, y2 = self.nodes[node2]
            L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            c = (x2 - x1) / L  # cos
            s = (y2 - y1) / L  # sin
            k = element['E'] * element['A'] / L
            # 局部到全局的坐标变换

            T = np.array([[c, s, 0, 0],
                          [0, 0, c, s]])

            # 局部刚度矩阵
            k_local = k * np.array([[1, -1],
                                    [-1, 1]])
            # 全局刚度矩阵
            k_global = T.T @ k_local @ T

            indices = [2 * node1 -2, 2 * node1 - 1, 2 * node2 -2, 2 * node2 - 1]
            for i in range(4):
                for j in range(4):
                    self.K[indices[i], indices[j]] += k_global[i, j]

    def apply_boundary_conditions(self,node,state):   # 3种状态 0：全固定 1：固定x 2：固定y
        """ 应用边界条件（固定第一个节点） """
        indices_to_remove = []
        if state == 0:  # 全固定
            indices_to_remove += [2 * (node - 1), 2 * node - 1]
        elif state == 1:  # 固定 x
            indices_to_remove += [2 * (node - 1)]
        elif state == 2:  # 固定 y
            indices_to_remove += [2 * node - 1]

        # 将固定的自由度添加到边界列表中，用于之后从力矩阵中删除
        self.boundary.extend(indices_to_remove)

    def solved_K(self):

        self.K_mod = np.copy(self.K)
        self.K_mod = np.delete(self.K_mod,  self.boundary, axis=0)
        self.K_mod = np.delete(self.K_mod,  self.boundary, axis=1)

    def solve(self):
        """ 解决位移 """

        n_dof = 2 * self.node_count
        forces = np.zeros(n_dof)
        self.displacements = np.ones(n_dof)

        for node, (fx, fy) in self.loads.items():
            idx = 2 * (node - 1)
            forces[idx] = fx
            forces[idx+1] = fy

        forces_new = np.delete(forces, self.boundary, axis=0)

        result = np.linalg.solve(self.K_mod, forces_new)  # 解得到位移

        # 先确定位移为0的点 再依次填入result
        self.displacements[self.boundary] = 0
        index = 0
        for i in range(len(self.displacements)):
            if self.displacements[i] == 1:
                self.displacements[i] = result[index]
                index += 1


    def print_results(self):
        """ 打印节点位移结果 """
        print("Node Displacements:")
        for i in range(1,self.node_count+1):
            print(f"Node {i}: ({self.displacements[2*(i-1)]} m, {self.displacements[2*(i-1)+1]} m)")


truss = Truss2D()

# 添加节点，指定它们的坐标
truss.add_node(1, 0, 0)
truss.add_node(2, 0.4, 0)
truss.add_node(3, 0.4, 0.3)
truss.add_node(4, 0, 0.3)

# 添加杆件，指定连接的节点、材料的弹性模量和横截面积
truss.add_element(1, 2, 295e9, 100e-6)
truss.add_element(2, 3, 295e9, 100e-6)
truss.add_element(1, 3, 295e9, 100e-6)
truss.add_element(3, 4, 295e9, 100e-6)

# 添加外力
truss.add_load(3, 0, -25000)
truss.add_load(2, 20000, 0)

# 组装全局刚度矩阵
truss.assemble_stiffness()

# 应用边界条件
truss.apply_boundary_conditions(1, 0)  # 节点1全固定
truss.apply_boundary_conditions(4, 0)  # 节点2全固定
truss.apply_boundary_conditions(2, 2)

truss.solved_K()

# 解决位移
truss.solve()

# 打印节点位移结果
truss.print_results()


