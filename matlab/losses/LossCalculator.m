classdef LossCalculator
    % LossCalculator - 损失函数计算器(静态方法类)
    % 实现Deep Ritz方法的损失函数计算
    %
    % 静态方法:
    %   compute_energy_loss_mc - 蒙特卡洛积分能量损失
    %   compute_energy_loss_quad - 高斯求积能量损失
    %   compute_boundary_loss - 边界条件损失
    %   compute_total_loss - 总损失

    methods (Static)
        function loss = compute_energy_loss_mc(output, grad_output, source_term, radius)
            % 计算能量泛函损失(蒙特卡洛积分)
            % E(u) = 1/2 ∫|∇u|² dx - ∫f·u dx
            %
            % 参数:
            %   output - 模型输出 [N x 1]
            %   grad_output - 梯度 [N x 2]
            %   source_term - 源项 [N x 1]
            %   radius - 域半径
            %
            % 返回:
            %   loss - 能量损失(标量)

            % 验证输入
            N = size(output, 1);
            if size(grad_output, 1) ~= N || size(source_term, 1) ~= N
                error('输入维度不匹配');
            end

            % 能量项: 0.5 * |∇u|²
            grad_norm_sq = sum(grad_output.^2, 2);  % [N x 1]
            energy_term = 0.5 * grad_norm_sq;

            % 源项: -f * u
            source_term_integral = -source_term .* output;

            % 蒙特卡洛积分: 乘以域面积
            area = pi * radius^2;
            loss = mean(energy_term) * area + mean(source_term_integral) * area;
        end

        function loss = compute_energy_loss_quad(output, grad_output, source_term, weights)
            % 计算能量泛函损失(高斯求积)
            % E(u) = 1/2 ∫|∇u|² dx - ∫f·u dx
            %
            % 参数:
            %   output - 模型输��� [N x 1]
            %   grad_output - 梯度 [N x 2]
            %   source_term - 源项 [N x 1]
            %   weights - 求积权重 [N x 1]
            %
            % 返回:
            %   loss - 能量损失(标量)

            % 验证输入
            N = size(output, 1);
            if size(grad_output, 1) ~= N || size(source_term, 1) ~= N || size(weights, 1) ~= N
                error('输入维度不匹配');
            end

            % 能量项: 0.5 * |∇u|²
            grad_norm_sq = sum(grad_output.^2, 2);  % [N x 1]
            energy_term = 0.5 * grad_norm_sq;

            % 源项: -f * u
            source_term_integral = -source_term .* output;

            % 高斯求积: 加权求和
            loss = sum(energy_term .* weights) + sum(source_term_integral .* weights);
        end

        function loss = compute_boundary_loss(output_bdry, target_bdry, penalty, radius)
            % 计算边界条件损失
            % L_bdry = penalty * ∫(u - g)² ds
            %
            % 参数:
            %   output_bdry - 边界上的模型输出 [M x 1]
            %   target_bdry - 边界上的目标值 [M x 1]
            %   penalty - 惩罚系数
            %   radius - 域半径
            %
            % 返回:
            %   loss - 边界损失(标量)

            % 验证输入
            M = size(output_bdry, 1);
            if size(target_bdry, 1) ~= M
                error('边界输入维度不匹配');
            end

            % Dirichlet边界条件惩罚: (u - g)²
            boundary_penalty = (output_bdry - target_bdry).^2;

            % 边界积分: 乘以边界长度
            boundary_length = 2 * pi * radius;
            loss = mean(boundary_penalty) * penalty * boundary_length;
        end

        function total_loss = compute_total_loss(energy_loss, boundary_loss)
            % 计算总损失
            % L_total = L_energy + L_boundary
            %
            % 参数:
            %   energy_loss - 能量损失(标量)
            %   boundary_loss - 边界损失(标量)
            %
            % 返回:
            %   total_loss - 总损失(标量)

            total_loss = energy_loss + boundary_loss;
        end
    end
end
