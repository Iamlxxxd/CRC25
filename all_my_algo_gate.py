# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/7/10 14:48
@project:    CRC25
"""

import os
import sys
import time
import traceback
from copy import deepcopy
from multiprocessing import Process, Queue

import yaml
from logger_config import logger
from visual import visual_map_foil_modded

# 获取当前文件所在目录（即submission目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（即submission的父目录）
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # 将项目根目录加入路径

from src.solver.MipSolver import MipSolver
from config import Config
from src.solver.SearchSolver import SearchSolver
from src.utils.common_utils import set_seed

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

base_dir = os.path.abspath(current_dir)
out_path = os.path.join(base_dir, "my_demo", "output", "retest")

mip_visual_path = os.path.join(out_path, "mip")
search_visual_path = os.path.join(out_path, "search")
hybrid_visual_path = os.path.join(out_path, "hybrid")

from src.AlgoTimer import AlgoTimer
from logger_config import logger


def single_mip(route_name=None):
    timer = AlgoTimer(time.time())
    set_seed()

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    # 设置地图名
    if route_name is not None:
        yaml_config['paths']['route_name'] = route_name

    # 初始化DataLoader，传入base_dir
    config = Config(yaml_config, base_dir=base_dir)
    config.out_path = out_path

    config.load_from_yaml()

    solver = MipSolver(config, timer)
    solver.init_from_config()

    solver.init_model()
    solver.solve_model()

    solver.process_solution_from_model()
    map_df = solver.out_put_df
    op_list = solver.out_put_op_list

    if config.visual:
        os.makedirs(mip_visual_path, exist_ok=True)
        visual_map_foil_modded(solver.process_visual_data(), mip_visual_path, config.route_name)

    return {"map_df": map_df, "op_list": op_list, "error": (solver.route_error, solver.graph_error)}


def single_search(route_name=None):
    timer = AlgoTimer(time.time())
    set_seed()

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    # 设置地图名
    if route_name is not None:
        yaml_config['paths']['route_name'] = route_name

    # 初始化DataLoader，传入base_dir
    config = Config(yaml_config, base_dir=base_dir)
    config.out_path = out_path

    config.load_from_yaml()

    solver = SearchSolver(config, timer)
    solver.init_from_config()

    solver.do_solve()

    solver.process_solution_from_model()
    map_df = solver.out_put_df
    op_list = solver.out_put_op_list

    if config.visual:
        os.makedirs(search_visual_path, exist_ok=True)
        visual_map_foil_modded(solver.process_visual_data(), search_visual_path, config.route_name)

    return {"map_df": map_df, "op_list": op_list, "error": (solver.route_error, solver.graph_error),
            "experiments_dict": solver.experiments_statistics}


def single_hybrid(route_name=None):
    timer = AlgoTimer(time.time())
    set_seed()

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    # 设置地图名
    if route_name is not None:
        yaml_config['paths']['route_name'] = route_name

    # 初始化DataLoader，传入base_dir
    config = Config(yaml_config, base_dir=base_dir)
    config.out_path = out_path

    config.load_from_yaml()

    mip_solver = MipSolver(config, timer)
    mip_solver.init_from_config()

    mip_solver.init_model()
    mip_start = time.time()
    mip_solver.solve_model()
    mip_cost = time.time() - mip_start

    mip_solver.process_solution_from_model()

    if config.visual:
        os.makedirs(hybrid_visual_path, exist_ok=True)
        visual_map_foil_modded(mip_solver.process_visual_data(), hybrid_visual_path, config.route_name + "_mip")

    if (mip_solver.route_error <= 0 and mip_solver.graph_error <= 1):
        # 改一条边或者不改 已经是最优了  不需要继续搜了 搜也不可能搜到更好的

        experiments_statistics = dict()
        experiments_statistics['route_error'] = mip_solver.route_error
        experiments_statistics['graph_error'] = mip_solver.graph_error
        experiments_statistics['mip_cost'] = mip_cost
        experiments_statistics['final'] = "mip"
        return {"map_df": mip_solver.out_put_df, "op_list": mip_solver.out_put_op_list,
                "error": (mip_solver.route_error, mip_solver.graph_error), "experiments_dict": experiments_statistics}

    if timer.time_over_check():
        timer.time_to_start("mip over time")
        # mip超时了
        return {"map_df": mip_solver.out_put_df, "op_list": mip_solver.out_put_op_list,
                "error": (mip_solver.route_error, mip_solver.graph_error)}

    timer = AlgoTimer(time.time())
    search_solver = SearchSolver(config, timer)
    search_solver.init_from_other_solver(mip_solver)
    search_solver.do_solve()

    search_solver.process_solution_from_model()
    if config.visual:
        os.makedirs(hybrid_visual_path, exist_ok=True)
        visual_map_foil_modded(search_solver.process_visual_data(), hybrid_visual_path,
                               config.route_name)

    return_dict = dict()
    experiments_statistics = search_solver.experiments_statistics
    if (search_solver.route_error, search_solver.graph_error) <= (mip_solver.route_error, mip_solver.graph_error):

        return_dict = {"map_df": search_solver.out_put_df, "op_list": search_solver.out_put_op_list,
                       "error": (search_solver.route_error, search_solver.graph_error)}
        experiments_statistics['final'] = "search"

    else:
        return_dict = {"map_df": mip_solver.out_put_df, "op_list": mip_solver.out_put_op_list,
                       "error": (mip_solver.route_error, mip_solver.graph_error)}
        experiments_statistics['route_error'] = mip_solver.route_error
        experiments_statistics['graph_error'] = mip_solver.graph_error
        experiments_statistics['final'] = "mip"

    experiments_statistics['mip_cost'] = mip_cost
    return_dict.update({"experiments_dict": experiments_statistics})
    return return_dict


def single_mip_from_args(args):
    timer = AlgoTimer(time.time())
    args.out_path = mip_visual_path

    set_seed()
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    config = Config(yaml_config, base_dir=base_dir)
    config.load_from_args(args)

    config.out_path = out_path

    solver = MipSolver(config, timer)
    solver.init_from_config()

    solver.init_model()
    solver.solve_model()

    solver.process_solution_from_model()
    map_df = solver.out_put_df
    op_list = solver.out_put_op_list

    if config.visual:
        os.makedirs(mip_visual_path, exist_ok=True)
        visual_map_foil_modded(solver.process_visual_data(), mip_visual_path, config.route_name)

    return {"map_df": map_df, "op_list": op_list, "error": (solver.route_error, solver.graph_error)}


def single_search_from_args(args):
    timer = AlgoTimer(time.time())
    args.out_path = search_visual_path
    set_seed()

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    config = Config(yaml_config, base_dir=base_dir)
    config.load_from_args(args)

    config.out_path = out_path

    # config.load_from_yaml()

    solver = SearchSolver(config, timer)
    solver.init_from_config()

    solver.do_solve()

    solver.process_solution_from_model()
    map_df = solver.out_put_df
    op_list = solver.out_put_op_list

    if config.visual:
        os.makedirs(search_visual_path, exist_ok=True)
        visual_map_foil_modded(solver.process_visual_data(), search_visual_path, config.route_name)

    return {"map_df": map_df, "op_list": op_list, "error": (solver.route_error, solver.graph_error)}


def single_hybrid_from_args(args):
    timer = AlgoTimer(time.time())
    args.out_path = hybrid_visual_path

    set_seed()

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    config = Config(yaml_config, base_dir=base_dir)
    config.load_from_args(args)

    config.out_path = out_path

    # config.load_from_yaml()

    mip_solver = MipSolver(config, timer)
    mip_solver.init_from_config()

    mip_solver.init_model()
    mip_solver.solve_model()

    mip_solver.process_solution_from_model()

    if config.visual:
        os.makedirs(hybrid_visual_path, exist_ok=True)
        visual_map_foil_modded(mip_solver.process_visual_data(), hybrid_visual_path, config.route_name + "_mip")

    if (mip_solver.route_error <= 0 and mip_solver.graph_error <= 1):
        # 改一条边或者不改 已经是最优了  不需要继续搜了 搜也不可能搜到更好的

        return {"map_df": mip_solver.out_put_df, "op_list": mip_solver.out_put_op_list,
                "error": (mip_solver.route_error, mip_solver.graph_error)}

    if timer.time_over_check():
        timer.time_to_start("mip over time")
        # mip超时了
        return {"map_df": mip_solver.out_put_df, "op_list": mip_solver.out_put_op_list,
                "error": (mip_solver.route_error, mip_solver.graph_error)}

    search_solver = SearchSolver(config, timer)
    search_solver.init_from_other_solver(mip_solver)
    search_solver.do_solve()

    search_solver.process_solution_from_model()
    if config.visual:
        os.makedirs(hybrid_visual_path, exist_ok=True)
        visual_map_foil_modded(search_solver.process_visual_data(), hybrid_visual_path,
                               config.route_name)

    if (search_solver.route_error, search_solver.graph_error) <= (mip_solver.route_error, mip_solver.graph_error):

        return {"map_df": search_solver.out_put_df, "op_list": search_solver.out_put_op_list,
                "error": (search_solver.route_error, search_solver.graph_error)}
    else:
        return {"map_df": mip_solver.out_put_df, "op_list": mip_solver.out_put_op_list,
                "error": (mip_solver.route_error, mip_solver.graph_error)}


def _run_search_worker(args, result_queue):
    """搜索算法工作进程"""
    try:
        result_dict = single_search_from_args(args)
        result_queue.put(('search', result_dict))
    except Exception as e:
        logger.error(f"search error: {traceback.format_exc(limit=99999)}")
        result_queue.put(('search', None, str(e)))


def _run_hybrid_worker(args, result_queue):
    """混合算法工作进程"""
    try:
        result_dict = single_hybrid_from_args(args)
        result_queue.put(('hybrid', result_dict))
    except Exception as e:
        logger.error(f"hybrid error: {traceback.format_exc(limit=99999)}")
        result_queue.put(('hybrid', None, str(e)))


def _compare_results(search_result, hybrid_result):
    """比较两个结果的质量，返回更好的结果"""
    # 检查结果是否有效
    if search_result is None and hybrid_result is None:
        return None  # 两个都没结果
    elif search_result is None:
        return hybrid_result  # 只有result2有结果
    elif hybrid_result is None:
        return search_result  # 只有result1有结果

    search_error = search_result.get("error", (float("inf"), float("inf")))
    hybrid_error = hybrid_result.get("error", (float("inf"), float("inf")))

    if search_error <= hybrid_error:
        logger.info(f"best solution from search:{str(search_error)}")
        return search_result
    else:
        logger.info(f"best solution from hybrid:{str(hybrid_error)}")
        return hybrid_result


def multi_job(args):
    """
    多进程执行搜索和混合算法
    :param args: 算法参数
    :return: 最优的map_df和op_list
    """
    start_time = time.time()
    timeout = AlgoTimer.time_limit  # 使用AlgoTimer的超时时间

    # 创建结果队列
    result_queue = Queue()

    # 创建两个进程
    search_process = Process(target=_run_search_worker, args=(deepcopy(args), result_queue))
    hybrid_process = Process(target=_run_hybrid_worker, args=(args, result_queue))

    # 启动进程
    search_process.start()
    hybrid_process.start()

    # 收集结果
    results = {}
    collected_results = 0

    while collected_results < 2:
        current_time = time.time()
        remaining_time = timeout - (current_time - start_time)

        if remaining_time <= 0:
            logger.warning("multi_job 超时")
            break

        try:
            # 等待结果，使用剩余时间作为超时
            result = result_queue.get(timeout=min(remaining_time, 1.0))
            algo_type = result[0]

            if len(result) == 3:  # 有异常
                logger.error(f"{algo_type} 算法执行出错: {result[2]}")
                results[algo_type] = None
            else:
                results[algo_type] = result[1]

            collected_results += 1
            logger.info(f"{algo_type} 算法完成 costTime:{time.time() - start_time}")

        except:
            # todo 内部报错好像不会退出
            # 超时或其他异常，继续等待
            continue

    # 确保进程结束
    if search_process.is_alive():
        search_process.terminate()
        search_process.join(timeout=1.0)
    if hybrid_process.is_alive():
        hybrid_process.terminate()
        hybrid_process.join(timeout=1.0)

    # 比较结果
    search_result = results.get('search', None)
    hybrid_result = results.get('hybrid', None)

    return_result = _compare_results(search_result, hybrid_result)

    return return_result
