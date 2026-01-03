import torch
import torch.nn as nn
from core.trainer.base_trainer import BaseTrainer
from heat_trainer_hr import HeatTrainer # 继承部分基础功能
from trapezoid_sampler_hr import HRSampler

class HeatTrainerHR(HeatTrainer):
    def __init__(self, model, device: str, params: dict):
        # 初始化父类，但跳过父类的 _prepare_training_data，因为我们需要法向量
        # 所以这里我们手动初始化
        self.model = model
        self.device = device
        self.params = params
        self.data_dir = params.get("data_dir", "./output")
        
        # 定义两个优化器
        # T_params: 网络输出的第0维 (T) 相关的权重? 
        # 在全连接网络中很难完全解耦 T 和 q 的权重。
        # 简单做法：两个步骤更新同一个网络，或者使用两个独立的 head。
        # 这里假设使用你现有的 EnhancedRitzNet，它是一个整体。
        # 策略：我们使用两个优化器，都指向 model.parameters()，
        # 但在 Loss 计算时，通过 .detach() 来控制梯度流向。
        
        lr = params["optimizer"]["lr"]
        self.opt_T = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(self.model.parameters(), lr=lr) # q 可以给大一点的学习率
        
        self.scheduler_T = torch.optim.lr_scheduler.StepLR(self.opt_T, step_size=1000, gamma=0.9)
        self.scheduler_q = torch.optim.lr_scheduler.StepLR(self.opt_q, step_size=1000, gamma=0.9)

        self._prepare_training_data()

    def _prepare_training_data(self):
        """准备数据，包含法向量"""
        # 内部点 (不需要法向量)
        data_body_np = HRSampler.sample_domain(self.params["bodyBatch"])
        self.data_body = torch.from_numpy(data_body_np).float().to(self.device)
        self.data_body.requires_grad = True

        # 左边界 (Dirichlet, 需要法向量)
        coords_left, normals_left = HRSampler.sample_left_boundary_with_normal(self.params["bdryBatch"])
        self.data_left = torch.from_numpy(coords_left).float().to(self.device)
        self.normal_left = torch.from_numpy(normals_left).float().to(self.device)

        # 上边界 (Neumann, 原理上不需要法向量参与计算，因为直接给定了q_bar)
        # 但为了统一格式，我们还是生成一下
        coords_top, _ = HRSampler.sample_top_boundary_with_normal(self.params["bdryBatch"])
        self.data_top = torch.from_numpy(coords_top).float().to(self.device)

    def _divergence(self, q, x):
        """计算向量场 q 对 x 的散度"""
        u = q[:, 0:1]
        v = q[:, 1:2]
        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 0:1]
        dv_dy = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True, retain_graph=True)[0][:, 1:2]
        return du_dx + dv_dy

    def train_step_hr(self):
        """执行一步 HR 对抗训练"""
        
        k = self.params["physics"]["k_thermal"]
        s = self.params["physics"]["s_source"]
        T_bc = self.params["physics"]["T_left"]
        q_bc_top = self.params["physics"]["q_top"]
        area = self.params["trapezoid"]["area"]
        len_left = self.params["trapezoid"]["len_left"]
        len_top = self.params["trapezoid"]["len_top"]

        # ============================================
        # Step 1: Maximize q (Critic Update)
        # 固定 T，更新 q 以满足本构关系，并作为 Lagrange 乘子探测边界误差
        # ============================================
        self.opt_q.zero_grad()
        
        # 前向传播
        out_body = self.model(self.data_body)
        T_body, qx, qy = out_body[:, 0:1], out_body[:, 1:2], out_body[:, 2:3]
        q_vec = torch.cat([qx, qy], dim=1)
        
        # 计算散度 (需要导数)
        div_q = self._divergence(q_vec, self.data_body)
        
        # 1. 区域积分: -1/(2k)|q|^2 - T*(div(q) - s)
        # 注意：我们要更新 q，所以 T 视为常数 (.detach())
        term_energy = -1.0/(2*k) * (qx**2 + qy**2)
        term_equilibrium = -T_body.detach() * (div_q - s) # 符号注意：HR泛函定义差异，这里采用 -T(div q - s)
        
        # 2. Dirichlet 边界项: Integral [ (q.n) * (T - T_bar) ]
        # 左边界 x=0, n=(-1,0), q.n = -qx
        out_left = self.model(self.data_left)
        T_left, qx_left, _ = out_left[:, 0:1], out_left[:, 1:2], out_left[:, 2:3]
        
        # q.n = qx_left * normal_left_x + ... = qx_left * (-1)
        q_dot_n = qx_left * self.normal_left[:, 0:1] + out_left[:, 2:3] * self.normal_left[:, 1:2]
        
        # 这里的 T_left 也要 detach，因为这一步只优化 q
        term_boundary_D = q_dot_n * (T_left.detach() - T_bc)
        
        # 构建 Loss (我们要 Maximize J, 即 Minimize -J)
        J_functional = torch.mean(term_energy + term_equilibrium) * area + \
                       torch.mean(term_boundary_D) * len_left
        
        loss_q = -J_functional
        loss_q.backward()
        self.opt_q.step()
        self.scheduler_q.step()

        # ============================================
        # Step 2: Minimize T (Actor Update)
        # 固定 q，更新 T 以满足平衡和边界约束
        # ============================================
        self.opt_T.zero_grad()
        
        # 重新前向传播 (因为图在backward后释放了，且需要新的梯度路径)
        out_body = self.model(self.data_body)
        T_body, qx, qy = out_body[:, 0:1], out_body[:, 1:2], out_body[:, 2:3]
        q_vec = torch.cat([qx, qy], dim=1)
        
        # 计算散度 (q 视为常数，但 div_q 计算依赖图，这里 div_q 不需要梯度传回 q，只需要传回 x? 
        # 不，其实这一项 T * div_q，对 T 求导时 div_q 是系数。所以 detach 即可)
        div_q = self._divergence(q_vec, self.data_body).detach()
        
        # 1. 区域积分
        # -1/(2k)|q|^2 对 T 无梯度，忽略
        # -T * (div(q) - s)
        term_equilibrium = -T_body * (div_q - s)
        
        # 2. Dirichlet 边界项
        # Integral [ (q.n) * (T - T_bar) ]
        # q 固定 (detach)
        out_left = self.model(self.data_left)
        T_left, qx_left, _ = out_left[:, 0:1], out_left[:, 1:2], out_left[:, 2:3]
        q_dot_n = (qx_left * self.normal_left[:, 0:1]).detach() # 只取 q 的值
        
        term_boundary_D = q_dot_n * (T_left - T_bc)
        
        # 3. Neumann 边界项 (自然边界)
        # Integral [ -T * q_bar ] 
        # 注意符号：Flux + term means inflow? HR principles usually: - int T * q_bar
        out_top = self.model(self.data_top)
        T_top = out_top[:, 0:1]
        term_boundary_N = -T_top * q_bc_top
        
        # 构建 Loss (Minimize J)
        loss_T = torch.mean(term_equilibrium) * area + \
                 torch.mean(term_boundary_D) * len_left + \
                 torch.mean(term_boundary_N) * len_top
                 
        loss_T.backward()
        self.opt_T.step()
        self.scheduler_T.step()
        
        return loss_q.item(), loss_T.item()

    def train(self):
        """覆盖父类的 train 方法"""
        print("开始 HR 对抗训练 (Adversarial Training)...")
        self.model.train()
        
        # ... (日志文件代码略，同父类)
        
        for step in range(self.params["trainStep"]):
            # 执行一步 HR 训练
            l_q, l_t = self.train_step_hr()
            
            if step % self.params["writeStep"] == 0:
                print(f"Step {step}: Loss_q(Max)={-l_q:.4f}, Loss_T(Min)={l_t:.4f}")
                # 注意：HR Loss 不代表误差，只代表能量泛函值
            
            # 重新采样逻辑同父类...
            if step % self.params.get("sampleStep", 200) == 0 and step > 0:
                 self._prepare_training_data()
                 
        print("训练完成")