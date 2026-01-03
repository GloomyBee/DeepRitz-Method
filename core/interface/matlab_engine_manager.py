"""
MATLAB Engine Manager - 管理MATLAB引擎的生命周期
"""

import logging
from typing import Any, Optional
import time

logger = logging.getLogger(__name__)


class MatlabEngineManager:
    """MATLAB引擎管理器类"""

    def __init__(self, matlab_path: Optional[str] = None, startup_timeout: int = 30):
        """
        初始化MATLAB引擎管理器

        Args:
            matlab_path: MATLAB安装路径(可选)
            startup_timeout: 启动超时时间(秒)
        """
        self.matlab_path = matlab_path
        self.startup_timeout = startup_timeout
        self.engine = None
        self._is_running = False

    def start_engine(self) -> None:
        """
        启动MATLAB引擎

        Raises:
            ImportError: 如果matlab.engine未安装
            RuntimeError: 如果引擎启动失败
        """
        if self._is_running:
            logger.warning("MATLAB引擎已经在运行")
            return

        try:
            import matlab.engine
        except ImportError as e:
            error_msg = (
                "无法导入matlab.engine。请确保已安装MATLAB Engine API for Python。\n"
                "安装方法: cd \"matlabroot/extern/engines/python\" && python setup.py install"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e

        try:
            logger.info(f"正在启动MATLAB引擎(超时: {self.startup_timeout}秒)...")
            start_time = time.time()

            self.engine = matlab.engine.start_matlab()

            elapsed = time.time() - start_time
            logger.info(f"MATLAB引擎启动成功(耗时: {elapsed:.2f}秒)")

            # 添加MATLAB路径
            if self.matlab_path:
                self.engine.addpath(self.matlab_path, nargout=0)
                logger.info(f"已添加MATLAB路径: {self.matlab_path}")

            # 添加项目MATLAB目录到路径
            self.engine.addpath('matlab/pdes', nargout=0)
            self.engine.addpath('matlab/losses', nargout=0)
            self.engine.addpath('matlab/sampling', nargout=0)
            self.engine.addpath('matlab/visualization', nargout=0)
            logger.info("已添加项目MATLAB目录到路径")

            self._is_running = True

        except Exception as e:
            error_msg = f"MATLAB引擎启动失败: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def stop_engine(self) -> None:
        """
        停止MATLAB引擎
        """
        if not self._is_running or self.engine is None:
            logger.warning("MATLAB引擎未运行")
            return

        try:
            logger.info("正在停止MATLAB引擎...")
            self.engine.quit()
            self.engine = None
            self._is_running = False
            logger.info("MATLAB引擎已停止")
        except Exception as e:
            logger.error(f"停止MATLAB引擎时出错: {str(e)}")
            self.engine = None
            self._is_running = False

    def call_function(self, func_name: str, *args, **kwargs) -> Any:
        """
        调用MATLAB函数

        Args:
            func_name: MATLAB函数名(支持类方法,如'Sampler.sample_from_disk')
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            MATLAB函数返回值

        Raises:
            RuntimeError: 如果引擎未运行或函数调用失败
        """
        if not self._is_running or self.engine is None:
            raise RuntimeError("MATLAB引擎未运行,请先调用start_engine()")

        try:
            logger.debug(f"调用MATLAB函数: {func_name}")

            # 处理类方法调用(如Sampler.sample_from_disk)
            if '.' in func_name:
                # 使用eval调用类方法
                result = self.engine.eval(
                    f"{func_name}({','.join(map(str, args))})",
                    nargout=kwargs.get('nargout', 1)
                )
            else:
                # 直接调用函数
                func = getattr(self.engine, func_name)
                result = func(*args, **kwargs)

            logger.debug(f"MATLAB函数调用成功: {func_name}")
            return result

        except AttributeError as e:
            error_msg = f"MATLAB函数不存在: {func_name}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"MATLAB函数调用失败: {func_name}, 错误: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def is_running(self) -> bool:
        """
        检查MATLAB引擎是否正在运行

        Returns:
            True如果引擎正在运行,否则False
        """
        return self._is_running and self.engine is not None

    def __enter__(self):
        """上下文管理器入口"""
        self.start_engine()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_engine()

    def __del__(self):
        """析构函数,确保引擎被关闭"""
        if self._is_running:
            self.stop_engine()
