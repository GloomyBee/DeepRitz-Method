% test_Sampler - 测试Sampler类
% 验证采样功能的正确性

function test_Sampler()
    fprintf('测试 Sampler 类...\n');

    radius = 1.0;
    num_points = 1000;

    % 测试圆盘内部采样
    fprintf('  测试圆盘内部采样...\n');
    points = Sampler.sample_from_disk(radius, num_points);
    assert(size(points, 1) == num_points, '采样点数不正确');
    assert(size(points, 2) == 2, '采样点维度不正确');

    % 验证所有点都在圆盘内
    distances = sqrt(sum(points.^2, 2));
    assert(all(distances <= radius + 1e-10), '存在圆盘外的点');
    fprintf('    通过\n');

    % 测试边界采样
    fprintf('  测试边界采样...\n');
    boundary_points = Sampler.sample_from_surface(radius, 100);
    assert(size(boundary_points, 1) == 100, '边界采样点数不正确');

    % 验证所有点都在边界上
    boundary_distances = sqrt(sum(boundary_points.^2, 2));
    assert(all(abs(boundary_distances - radius) < 1e-10), '边界点不在圆周上');
    fprintf('    通过\n');

    % 测试高斯求积
    fprintf('  测试高斯求积...\n');
    [quad_points, weights] = Sampler.gauss_quadrature_2d(3);
    assert(size(quad_points, 1) == 9, '求积点数不正确');  % 3x3 = 9
    assert(size(weights, 1) == 9, '权重数不正确');
    assert(abs(sum(weights) - 4.0) < 1e-10, '权重和不正确');  % [-1,1]x[-1,1] 面积为4
    fprintf('    通过\n');

    % 测试网格生成
    fprintf('  测试网格生成...\n');
    grid = Sampler.generate_test_grid(radius, 50);
    assert(size(grid, 2) == 2, '网格维度不正确');

    % 验证所有点都在圆盘内
    grid_distances = sqrt(sum(grid.^2, 2));
    assert(all(grid_distances <= radius + 1e-10), '网格存在圆盘外的点');
    fprintf('    通过\n');

    fprintf('Sampler 测试全部通过!\n\n');
end
