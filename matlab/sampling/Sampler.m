classdef Sampler
    % Sampler - 采样工具类(静态方法类)
    % 提供各种采样方法
    %
    % 静态方法:
    %   sample_from_disk - 圆盘内部均匀采样
    %   sample_from_surface - 圆盘边界采样
    %   gauss_quadrature_2d - 2D高斯求积点和权重
    %   generate_test_grid - 生成测试网格

    methods (Static)
        function points = sample_from_disk(radius, num_points)
            % 在圆盘内部均匀采样
            % 使用极坐标变换: r = sqrt(uniform) * radius, θ = uniform * 2π
            %
            % 参数:
            %   radius - 圆盘半径
            %   num_points - 采样点数
            %
            % 返回:
            %   points - 采样点坐标 [num_points x 2]

            % 生成随机半径和角度
            r = sqrt(rand(num_points, 1)) * radius;
            theta = rand(num_points, 1) * 2 * pi;

            % 转换为笛卡尔坐标
            x = r .* cos(theta);
            y = r .* sin(theta);

            points = [x, y];
        end

        function points = sample_from_surface(radius, num_points)
            % 在圆盘边界上均匀采样
            %
            % 参数:
            %   radius - 圆盘半径
            %   num_points - 采样点数
            %
            % 返回:
            %   points - 边界采样点坐标 [num_points x 2]

            % 均匀分布的角度
            theta = linspace(0, 2*pi, num_points+1);
            theta = theta(1:end-1);  % 去掉最后一个点(与第一个点重合)

            % 转换为笛卡尔坐标
            x = radius * cos(theta');
            y = radius * sin(theta');

            points = [x, y];
        end

        function [points, weights] = gauss_quadrature_2d(order)
            % 生成2D高斯求积点和权重
            % 使用张量积方法
            %
            % 参数:
            %   order - 求积阶数
            %
            % 返回:
            %   points - 求积点坐标 [N x 2]
            %   weights - 求积权重 [N x 1]

            % 1D高斯-勒让德求积点和权重
            [x1d, w1d] = Sampler.gauss_legendre_1d(order);

            % 张量积生成2D点和权重
            [X, Y] = meshgrid(x1d, x1d);
            [WX, WY] = meshgrid(w1d, w1d);

            points = [X(:), Y(:)];
            weights = WX(:) .* WY(:);
        end

        function grid = generate_test_grid(radius, num_points)
            % 生成均匀测试网格
            %
            % 参数:
            %   radius - 圆盘半径
            %   num_points - 每个维度的点数
            %
            % 返回:
            %   grid - 网格点坐标 [N x 2] (仅包含圆盘内的点)

            % 生成均匀网格
            x = linspace(-radius, radius, num_points);
            y = linspace(-radius, radius, num_points);
            [X, Y] = meshgrid(x, y);

            % 只保留圆盘内的点
            mask = X.^2 + Y.^2 <= radius^2;
            grid = [X(mask), Y(mask)];
        end

        function [x, w] = gauss_legendre_1d(n)
            % 计算1D高斯-勒让德求积点和权重
            % 区间 [-1, 1]
            %
            % 参数:
            %   n - 求积点数
            %
            % 返回:
            %   x - 求积点 [n x 1]
            %   w - 权重 [n x 1]

            % 使用MATLAB内置函数或自定义实现
            % 这里使用简化的实现

            % 初始猜测
            i = 1:n;
            x = cos(pi * (i - 0.25) / (n + 0.5));

            % 牛顿迭代
            for iter = 1:10
                [P, Pp] = Sampler.legendre_poly(n, x);
                x = x - P ./ Pp;
            end

            % 计算权重
            [~, Pp] = Sampler.legendre_poly(n, x);
            w = 2 ./ ((1 - x.^2) .* Pp.^2);

            x = x';
            w = w';
        end

        function [P, Pp] = legendre_poly(n, x)
            % 计算勒让德多项式及其导数
            %
            % 参数:
            %   n - 多项式阶数
            %   x - 评估点
            %
            % 返回:
            %   P - 多项式值
            %   Pp - 导数值

            P0 = ones(size(x));
            P1 = x;

            for k = 2:n
                P = ((2*k-1) * x .* P1 - (k-1) * P0) / k;
                P0 = P1;
                P1 = P;
            end

            Pp = n * (x .* P - P0) ./ (x.^2 - 1);
        end
    end
end
