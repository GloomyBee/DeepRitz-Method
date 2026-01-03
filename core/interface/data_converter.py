"""
Data Converter - Python和MATLAB数据格式转换
"""

import torch
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)


class DataConverter:
    """Python和MATLAB数据格式转换器"""

    @staticmethod
    def torch_to_matlab(tensor: torch.Tensor):
        """
        将PyTorch张量转换为MATLAB数组

        Args:
            tensor: PyTorch张量

        Returns:
            MATLAB double数组

        Raises:
            ImportError: 如果matlab.engine未安装
            ValueError: 如果输入无效
        """
        try:
            import matlab
        except ImportError as e:
            raise ImportError("无法导入matlab模块,请安装MATLAB Engine API for Python") from e

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"输入必须是torch.Tensor,得到: {type(tensor)}")

        # 转换为NumPy数组(在CPU上)
        if tensor.is_cuda:
            numpy_array = tensor.detach().cpu().numpy()
        else:
            numpy_array = tensor.detach().numpy()

        # 转换为MATLAB数组
        return DataConverter.numpy_to_matlab(numpy_array)

    @staticmethod
    def numpy_to_matlab(array: np.ndarray):
        """
        将NumPy数组转换为MATLAB数组

        Args:
            array: NumPy数组

        Returns:
            MATLAB double数组

        Raises:
            ImportError: 如果matlab.engine未安装
            ValueError: 如果输入无效
        """
        try:
            import matlab
        except ImportError as e:
            raise ImportError("无法导入matlab模块,请安装MATLAB Engine API for Python") from e

        if not isinstance(array, np.ndarray):
            raise ValueError(f"输入必须是np.ndarray,得到: {type(array)}")

        # 确保数组是连续的
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        # 转换为float64(MATLAB double)
        if array.dtype != np.float64:
            array = array.astype(np.float64)

        # 转换为MATLAB数组
        # MATLAB使用列优先(Fortran)顺序,需要转置
        if array.ndim == 1:
            # 1D数组转换为列向量
            matlab_array = matlab.double(array.tolist())
        elif array.ndim == 2:
            # 2D数组需要转置
            matlab_array = matlab.double(array.tolist())
        else:
            # 高维数组
            matlab_array = matlab.double(array.tolist())

        logger.debug(f"NumPy数组转换为MATLAB: shape={array.shape}, dtype={array.dtype}")
        return matlab_array

    @staticmethod
    def matlab_to_torch(data, device: str = 'cpu') -> torch.Tensor:
        """
        将MATLAB数组转换为PyTorch张量

        Args:
            data: MATLAB数组
            device: 目标设备('cpu'或'cuda')

        Returns:
            PyTorch张量

        Raises:
            ValueError: 如果输入无效
        """
        # 先转换为NumPy
        numpy_array = DataConverter.matlab_to_numpy(data)

        # 转换为PyTorch张量
        tensor = torch.from_numpy(numpy_array).float()

        # 移动到指定设备
        if device != 'cpu':
            tensor = tensor.to(device)

        logger.debug(f"MATLAB数组转换为PyTorch: shape={tensor.shape}, device={tensor.device}")
        return tensor

    @staticmethod
    def matlab_to_numpy(data) -> np.ndarray:
        """
        将MATLAB数组转换为NumPy数组

        Args:
            data: MATLAB数组

        Returns:
            NumPy数组

        Raises:
            ValueError: 如果输入无效
        """
        try:
            import matlab
        except ImportError as e:
            raise ImportError("无法导入matlab模块,请安装MATLAB Engine API for Python") from e

        if data is None:
            raise ValueError("MATLAB数据为None")

        # 检查是否是MATLAB数组类型
        if isinstance(data, (matlab.double, matlab.single, matlab.int32, matlab.int64)):
            # 转换为NumPy数组
            numpy_array = np.array(data)

            # MATLAB使用列优先,可能需要调整形状
            # 如果是2D数组,MATLAB的行列与NumPy相反
            if numpy_array.ndim == 2 and numpy_array.shape[0] == 1:
                # 行向量,转换为1D数组
                numpy_array = numpy_array.flatten()
            elif numpy_array.ndim == 2:
                # 2D数组,保持原样(MATLAB的列优先已经被np.array处理)
                pass

            logger.debug(f"MATLAB数组转换为NumPy: shape={numpy_array.shape}, dtype={numpy_array.dtype}")
            return numpy_array

        elif isinstance(data, (list, tuple)):
            # 如果是Python列表或元组,直接转换
            return np.array(data)

        elif isinstance(data, (int, float)):
            # 标量值
            return np.array([data])

        else:
            raise ValueError(f"不支持的MATLAB数据类型: {type(data)}")

    @staticmethod
    def validate_shape(array: Union[torch.Tensor, np.ndarray], expected_shape: tuple) -> bool:
        """
        验证数组形状

        Args:
            array: 要验证的数组
            expected_shape: 期望的形状(可以使用None表示任意维度)

        Returns:
            True如果形状匹配,否则False
        """
        if isinstance(array, torch.Tensor):
            actual_shape = array.shape
        elif isinstance(array, np.ndarray):
            actual_shape = array.shape
        else:
            return False

        if len(actual_shape) != len(expected_shape):
            return False

        for actual, expected in zip(actual_shape, expected_shape):
            if expected is not None and actual != expected:
                return False

        return True

    @staticmethod
    def ensure_2d(array: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        确保数组是2D的

        Args:
            array: 输入数组

        Returns:
            2D数组
        """
        if isinstance(array, torch.Tensor):
            if array.ndim == 1:
                return array.unsqueeze(1)
            elif array.ndim == 2:
                return array
            else:
                raise ValueError(f"无法将{array.ndim}D数组转换为2D")
        elif isinstance(array, np.ndarray):
            if array.ndim == 1:
                return array.reshape(-1, 1)
            elif array.ndim == 2:
                return array
            else:
                raise ValueError(f"无法将{array.ndim}D数组转换为2D")
        else:
            raise ValueError(f"不支持的数组类型: {type(array)}")
