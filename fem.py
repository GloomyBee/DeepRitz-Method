import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class HeatFEM:
    def __init__(self):
        # ==========================================
        # 1. 物理参数 (对应上传的图 7.16 和手算过程)
        # ==========================================
        self.k = 20.0  # 导热系数 W/(m·C)
        self.s = 50.0  # 热源 W/m³
        self.t = 0.1  # 厚度 m (注意：刚度矩阵和载荷向量都要乘厚度)
        self.T_left = 20.0  # Dirichlet边界: x=0, T=20
        self.q_top = 100.0  # Neumann边界: y=1, q_n = -100 (流入)
        # 注意：FEM载荷向量通常写为 +∫N*q_in dl
        # 图中箭头向下流入，物理上表示加热，
        # 所以这里载荷项应该是正的 (q_flux = 100)

    def generate_mesh(self, nx=51, ny=51):
        """生成梯形区域的三角网格"""
        # 生成矩形点阵
        x = np.linspace(0, 2, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])

        # 筛选梯形内部点: x + y <= 2.0
        # 考虑到浮点误差，加一个微小量
        mask = (points[:, 0] + points[:, 1] <= 2.0 + 1e-8)
        self.nodes = points[mask]
        self.n_nodes = len(self.nodes)

        # Delaunay 三角剖分
        self.tri = Delaunay(self.nodes)
        self.elems = self.tri.simplices
        self.n_elems = len(self.elems)

        print(f"网格生成完成: 节点数 {self.n_nodes}, 单元数 {self.n_elems}")

    def assemble(self):
        """组装全局刚度矩阵 K 和 载荷向量 F"""
        # 使用 LIL 格式方便逐步填充，最后转 CSR 求解
        K = sp.lil_matrix((self.n_nodes, self.n_nodes))
        F = np.zeros(self.n_nodes)

        # D 矩阵 (各向同性)
        D_mat = np.array([[self.k, 0], [0, self.k]])

        # --- 1. 循环遍历单元进行组装 ---
        for el_idx, node_indices in enumerate(self.elems):
            # 获取单元的三个节点坐标
            coords = self.nodes[node_indices]  # shape (3, 2)
            x, y = coords[:, 0], coords[:, 1]

            # 计算面积 Area = 0.5 * det(J)
            # b_i = y_j - y_m
            # c_i = x_m - x_j
            b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
            c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

            TwoA = x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
            if np.abs(TwoA) < 1e-14:
                continue
            Area = 0.5 * np.abs(TwoA)

            # 几何矩阵 B (常应变三角形 CST)
            # B = [dN/dx; dN/dy] = 1/2A * [b; c]
            B = (1.0 / TwoA) * np.vstack([b, c])

            # 单元刚度矩阵 Ke = t * Area * B.T * D * B
            # 你的手算图片 image_68eb62.png 中包含了 t (thickness)
            Ke = self.t * Area * np.dot(B.T, np.dot(D_mat, B))

            # 单元体载荷 Fe = t * Area * s * [1/3, 1/3, 1/3]
            # 对应手算 image_68eb62.png Body force
            Fe = (self.t * Area * self.s / 3.0) * np.ones(3)

            # 填入全局矩阵
            for i in range(3):
                row = node_indices[i]
                F[row] += Fe[i]
                for j in range(3):
                    col = node_indices[j]
                    K[row, col] += Ke[i, j]

        # --- 2. 处理 Neumann 边界 (上边界 y=1) ---
        # 对应手算 image_68eba6.png Traction
        tol = 1e-6

        # Robust boundary-edge extraction: apply Neumann only on true boundary edges (used by exactly 1 triangle).
        # This avoids double-counting when Delaunay drops collinear hull points on y=1.
        edge_counts = {}
        for tri_nodes in self.elems:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                i1 = int(tri_nodes[a])
                i2 = int(tri_nodes[b])
                key = (i1, i2) if i1 < i2 else (i2, i1)
                edge_counts[key] = edge_counts.get(key, 0) + 1

        for (n1_idx, n2_idx), cnt in edge_counts.items():
            if cnt != 1:
                continue

            p1 = self.nodes[n1_idx]
            p2 = self.nodes[n2_idx]

            # Fig 7.16: Neumann only on top edge y=1 and x in [0, 1]
            if (np.abs(p1[1] - 1.0) < tol and np.abs(p2[1] - 1.0) < tol):
                if p1[0] <= 1.0 + tol and p2[0] <= 1.0 + tol:
                    length = np.linalg.norm(p1 - p2)
                    # self.q_top here is the weak-form flux density g = k*dT/dn (so physical q·n = -g).
                    load = self.t * self.q_top * length / 2.0
                    F[n1_idx] += load
                    F[n2_idx] += load

        for i in range(0):
            tri_nodes = self.elems[i]
            # 检查三角形的三条边
            edges = [(0, 1), (1, 2), (2, 0)]
            for e_local in edges:
                n1_idx = tri_nodes[e_local[0]]
                n2_idx = tri_nodes[e_local[1]]
                p1 = self.nodes[n1_idx]
                p2 = self.nodes[n2_idx]

                # 判断是否在上边界 (y=1 且 x<=1)
                # 注意图7.16中，Neumann边界只在 (0,1) 到 (1,1) 这一段
                if (np.abs(p1[1] - 1.0) < tol and np.abs(p2[1] - 1.0) < tol):
                    # 确保不在斜边上 (虽然 x+y<=2 在 y=1时 x<=1，为了保险)
                    if p1[0] <= 1.0 + tol and p2[0] <= 1.0 + tol:
                        length = np.linalg.norm(p1 - p2)
                        # 边载荷 = t * q * L / 2 分配到两个节点
                        # q_top 为正值(流入)，增加节点能量
                        load = self.t * self.q_top * length / 2.0
                        F[n1_idx] += load
                        F[n2_idx] += load

        # --- 3. 处理 Dirichlet 边界 (左边界 x=0) ---
        # 使用“大数置1法”或“划行划列法”，这里用置1法保持矩阵大小不变
        left_indices = np.where(self.nodes[:, 0] < tol)[0]

        for idx in left_indices:
            # 将该行清零
            K[idx, :] = 0.0
            # 对角线设为1
            K[idx, idx] = 1.0
            # 右端项设为指定温度
            F[idx] = self.T_left

        self.K = K.tocsr()
        self.F = F

    def solve(self):
        print("开始求解线性方程组...")
        self.U = spla.spsolve(self.K, self.F)
        print("求解完成.")
        return self.U

    def plot_result(self):
        # 创建三角剖分对象用于绘图
        triang = mtri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elems)

        plt.figure(figsize=(10, 8))
        # 绘制温度云图
        tpc = plt.tripcolor(triang, self.U, shading='gouraud', cmap='inferno')
        plt.colorbar(tpc, label='Temperature (°C)')

        # 绘制网格线 (可选)
        plt.triplot(triang, 'k-', alpha=0.1, linewidth=0.5)

        # 设置图形属性
        plt.title(f"FEM Reference Solution (Nodes: {self.n_nodes})", fontsize=14)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.xlim(-0.1, 2.1)
        plt.ylim(-0.1, 1.1)

        # 标注特定点 (对比手算结果)
        # 寻找最接近 (1,0) 和 (1,1) 的节点
        # 手算结果: Node 2 (1,0) -> 26.23, Node 4 (1,1) -> 24.60
        # 注意：手算的坐标定义可能不同，根据图7.16：
        # 节点2是(2,0)？不，图7.16中节点2在最右下角。
        # 你的手算图 image_68eb69.png 显示 Element 2 的节点是 (0,0), (1,0), (1,1)。
        # 所以手算中的 T2 是 (1,0)，T3是(1,1)。
        # 让我们找 (1,0) 和 (1,1) 的值。

        target_pts = [(1.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        labels = ["(1,0)", "(1,1)", "(2,0)"]

        print("\n=== 关键点温度对比 ===")
        for pt, label in zip(target_pts, labels):
            dists = np.linalg.norm(self.nodes - np.array(pt), axis=1)
            idx = np.argmin(dists)
            val = self.U[idx]
            plt.plot(pt[0], pt[1], 'ro')
            plt.text(pt[0] + 0.05, pt[1], f"{val:.2f}°C", color='blue', fontweight='bold')
            print(f"Point {label}: {val:.4f} °C")

        plt.savefig('fem_solution.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    fem = HeatFEM()
    # 使用较密的网格以获得收敛的“真值”
    fem.generate_mesh(nx=101, ny=51)
    fem.assemble()
    fem.solve()
    fem.plot_result()
