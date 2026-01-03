classdef Poisson2D
    % Poisson2D - 二维泊松方程类
    % 求解 -Δu = f 在圆域上的问题
    %
    % 属性:
    %   radius - 圆域半径
    %
    % 方法:
    %   source_term - 计算源项 f(x,y) = 4
    %   exact_solution - 计算解析解 u(x,y) = 1 - x^2 - y^2
    %   boundary_condition - 计算边界条件
    %   exact_gradient - 计算解析解的梯度

    properties
        radius  % 圆域半径
    end

    methods
        function obj = Poisson2D(radius)
            % 构造函数
            %
            % 参数:
            %   radius - 圆域半径 (默认: 1.0)

            if nargin < 1
                radius = 1.0;
            end
            obj.radius = radius;
        end

        function f = source_term(obj, points)
            % 计算源项 f(x, y) = 4
            %
            % 参数:
            %   points - 输入点坐标 [N x 2] 矩阵
            %
            % 返回:
            %   f - 源项值 [N x 1] 向量

            % 验证输入
            if size(points, 2) ~= 2
                error('输入点必须是 N x 2 矩阵');
            end

            N = size(points, 1);
            f = 4.0 * ones(N, 1);
        end

        function u = exact_solution(obj, points)
            % 计算解析解 u(x, y) = 1 - x^2 - y^2
            %
            % 参数:
            %   points - 输入点坐标 [N x 2] 矩阵
            %
            % 返回:
            %   u - 解析解值 [N x 1] 向量

            % 验证输入
            if size(points, 2) ~= 2
                error('输入点必须是 N x 2 矩阵');
            end

            x = points(:, 1);
            y = points(:, 2);
            u = 1.0 - x.^2 - y.^2;
        end

        function g = boundary_condition(obj, points)
            % 计算边界条件 (Dirichlet边界条件)
            % 边界上 u = exact_solution
            %
            % 参数:
            %   points - 边界点坐标 [N x 2] 矩阵
            %
            % 返回:
            %   g - 边界条件值 [N x 1] 向量

            g = obj.exact_solution(points);
        end

        function grad_u = exact_gradient(obj, points)
            % 计算解析解的梯度 ∇u = [-2x, -2y]
            %
            % 参数:
            %   points - 输入点坐标 [N x 2] 矩阵
            %
            % 返回:
            %   grad_u - 梯度 [N x 2] 矩阵

            % 验证输入
            if size(points, 2) ~= 2
                error('输入点必须是 N x 2 矩阵');
            end

            x = points(:, 1);
            y = points(:, 2);
            grad_u = [-2.0 * x, -2.0 * y];
        end
    end
end
