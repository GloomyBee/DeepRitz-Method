% test_Poisson2D - 测试Poisson2D类
% 验证PDE定义的正确性

function test_Poisson2D()
    fprintf('测试 Poisson2D 类...\n');

    % 创建PDE实例
    pde = Poisson2D(1.0);

    % 测试点
    test_points = [0.0, 0.0; 0.5, 0.5; 1.0, 0.0];

    % 测试源项
    fprintf('  测试源项...\n');
    f = pde.source_term(test_points);
    expected_f = [4.0; 4.0; 4.0];
    assert(all(abs(f - expected_f) < 1e-10), '源项计算错误');
    fprintf('    通过\n');

    % 测试解析解
    fprintf('  测试解析解...\n');
    u = pde.exact_solution(test_points);
    expected_u = [1.0; 0.5; 0.0];
    assert(all(abs(u - expected_u) < 1e-10), '解析解计算错误');
    fprintf('    通过\n');

    % 测试边界条件
    fprintf('  测试边界条件...\n');
    boundary_points = [1.0, 0.0; 0.0, 1.0];
    g = pde.boundary_condition(boundary_points);
    expected_g = [0.0; 0.0];
    assert(all(abs(g - expected_g) < 1e-10), '边界条件计算错误');
    fprintf('    通过\n');

    % 测试梯度
    fprintf('  测试梯度...\n');
    grad_u = pde.exact_gradient(test_points);
    expected_grad = [0.0, 0.0; -1.0, -1.0; -2.0, 0.0];
    assert(all(all(abs(grad_u - expected_grad) < 1e-10)), '梯度计算错误');
    fprintf('    通过\n');

    fprintf('Poisson2D 测试全部通过!\n\n');
end
