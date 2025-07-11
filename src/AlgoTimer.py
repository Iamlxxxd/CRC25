import time

from logger_config import logger


class AlgoTimer:
    time_limit = 290  # 留10s写文件

    def __init__(self, start_time):
        """
        初始化计时器
        :param start_time: 开始时间，通常是time.time()的返回值
        """
        self.start_time = start_time
        self.last_check_time = start_time  # 上次检查点的开始时间，初始化为全局开始时间
        self.check_points = {}  # 存储检查点的时间

    def time_over_check(self):
        """
        检查当前时间和开始时间是否超过time_limit
        :return: True表示超时，False表示未超时
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        return elapsed_time > self.time_limit

    def check_point(self, *args):
        """
        输入不定长参数，打印当前时间到上一个开始时间的时间差
        第一次check_point使用到全局开始时间的时间差
        :param args: 不定长参数，类似print函数
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_check_time

        # 将所有参数转换为字符串并连接
        message = ' '.join(str(arg) for arg in args)
        logger.info(f"{message} 耗时: {elapsed_time:.3f}秒")

        # 更新上次检查点时间为当前时间
        self.last_check_time = current_time

    def time_to_start(self, *args):
        message = ' '.join(str(arg) for arg in args)
        elapsed_time = time.time() - self.start_time
        logger.info(f"{message} 距离开始: {elapsed_time:.3f}秒")
