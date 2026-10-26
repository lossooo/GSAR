"""
实现整个流程，包括多设备多应用的环境初始化、任务收集、轨迹生成
"""
import os
import time
import logging
import json
from contextlib import nullcontext, redirect_stdout, redirect_stderr

from ..common.device import Device
from .utils.adb_command import DeviceInitializer

from .env_pull import Env_Pull
from .env_push import Env_Push
from .collect_task import collect_task, complexify_task
from .collect_traj import collect_traj

from ..config import get_data_synthesis_config, resolve_path

config = get_data_synthesis_config()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def gui_init(
    device_serial: str,
    package_name: str,
    adb_port: int = 5037,
    max_iterations: int = 5,
    max_tasks: int = 4,
    max_breadth: int = 10,
    max_depth: int = 8,
    max_step: int = 20,
    add_step: int = 5,
    data_root_dir: str = "./data_test",
):
    """
    GUI 初始化函数

    :param device_serial: 设备序列号
    :param package_name: 应用包名
    :param adb_port: adb 端口号
    :param max_iterations: 最大迭代次数
    :param max_tasks: 每个页面生成的最大任务数
    :param max_breadth: 每个页面可选操作的最大数量
    :param max_depth: 每次迭代探索的最大深度
    :param max_step: 每个任务的最大执行步数
    :param add_step: 复杂任务增加的步数
    :param data_root_dir: 数据存储根目录
    """
    logger.info(f"Starting GUI initialization for device {device_serial} and package {package_name}...")
    # 迭代进行任务收集
    total_task_list = []  # 任务列表
    total_traj_list = []  # 轨迹列表
    pre_traj_list = []  # 上一轮次的轨迹列表

    env_pull_dir= os.path.join(data_root_dir, "data_env")
    collect_task_dir = os.path.join(data_root_dir, "data_task")
    os.makedirs(os.path.join(data_root_dir, "task_json"), exist_ok=True)
    task_path = os.path.join(data_root_dir, "task_json", f"task_{package_name}.json")
    collect_traj_dir = os.path.join(data_root_dir, "data_traj")
    os.makedirs(os.path.join(data_root_dir, "traj_json"), exist_ok=True)
    traj_path = os.path.join(data_root_dir, "traj_json", f"traj_{package_name}.json")

    # 支持断点收集
    env_id_checkpoint = None
    if os.path.exists(task_path):
        with open(task_path, "r", encoding="utf-8") as f:
            total_task_list = json.load(f)
    env_id_checkpoint = total_task_list[-1]["env_id"] if total_task_list else None
    iteration_checkpoint = total_task_list[-1]["iteration"] if total_task_list else 0

    if os.path.exists(traj_path):
        with open(traj_path, "r", encoding="utf-8") as f:
            total_traj_list = json.load(f)
        pre_traj_list = [item for item in total_traj_list if item["iteration"] == iteration_checkpoint]
    # 开始迭代进化
    for iteration in range(iteration_checkpoint, max_iterations):
        logger.info(f"\nIteration {iteration + 1}/{max_iterations}\n")

        # 1.拉取环境
        if not env_id_checkpoint:
            logger.info(f"Plulling environment for {package_name} in {device_serial}...")
            if iteration == 0: 
                # 第一次迭代，直接使用初始环境
                env_pull = Env_Pull(
                    root_dir=env_pull_dir,
                    adb_port=adb_port,
                    device_serial=device_serial,
                    package_name=package_name,
                )
                cur_env_id = env_pull.pull()
            else:   
                # 后续迭代，更新环境
                env_pull = Env_Pull(
                    root_dir=env_pull_dir,
                    adb_port=adb_port,
                    device_serial=device_serial,
                    package_name=package_name,
                )
                cur_env_id = env_pull.pull()            
        else:
            # 从检查点恢复进化
            logger.info(f"Resuming from checkpoint env_id({env_id_checkpoint})")
            cur_env_id = env_id_checkpoint
            env_id_checkpoint = None

            env_push = Env_Push(
                root_dir=env_pull_dir,
                adb_port=adb_port,
                device_serial=device_serial,
                package_name=package_name,
                env_id=cur_env_id,
            )
            env_push.push()

            # task_list = [task for task in total_task_list if task["env_id"] == cur_env_id]
        
            # traj_list_set = set()   # 已经执行过的轨迹集合
            # for traj in total_traj_list:
            #     traj_list_set.add(traj["query"])
            # task_list = [task for task in task_list if task["query"] not in traj_list_set]

        # 2.收集任务
        logger.info(f"Collecting tasks for {package_name} in {device_serial}...")
        task_list = collect_task(
            package_name=package_name,
            device_serial=device_serial,
            env_id=cur_env_id,
            max_tasks=max_tasks,
            max_breadth=max_breadth,
            max_depth=max_depth,
            iteration=iteration + 1,
            output_dir=collect_task_dir
        )

        # 根据上一轮的轨迹进行指令复杂化
        logger.info(f"Complexifying tasks based on previous trajectories for {package_name}...")
        if iteration > 0 and len(pre_traj_list) > 0: 
            for traj in pre_traj_list:
                step_datas = traj.get("step_datas") or []
                final_action = (step_datas[-1].get("action_output_json") or "") if step_datas else ""
                complete = "complete" in final_action
                if not traj.get("is_complexified", False) and complete:
                    complex_task = complexify_task(traj, cur_env_id)
                    task_list.extend(complex_task)

        total_task_list.extend(task_list)

        logger.info(f"Saving tasks for {package_name}...")
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(total_task_list, f, indent=4, ensure_ascii=False, default=str)

        # 3.上传环境
        logger.info(f"Pushing environment for {package_name} in {device_serial}...")
        env_push = Env_Push(
            root_dir=env_pull_dir,
            adb_port=adb_port,
            device_serial=device_serial,
            package_name=package_name,
            env_id=cur_env_id,
        )
        env_push.push()

        # 4.更新环境 & 收集轨迹
        logger.info(f"Collecting trajectories for {package_name} in {device_serial}...")
        traj_list = collect_traj(
            task_list=task_list,
            device_serial=device_serial,
            max_step=max_step,
            add_step=add_step,
            output_dir=collect_traj_dir,
        )
        total_traj_list.extend(traj_list)
        pre_traj_list = traj_list  # 更新上一轮的轨迹列表

        logger.info(f"Saving trajectories for {package_name}...")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(total_traj_list, f, indent=4, ensure_ascii=False, default=str)
    
    logger.info(f"GUI initialization for device {device_serial} and package {package_name} completed!")


def device_init(device_serial: str, package_name: str, adb_port: int = 5037):
    """
    设备初始化主函数：按顺序执行所有初始化步骤

    :device_serial: 设备序列号
    :package_name: 应用包名
    :adb_port: adb 端口
    """
    device_init = DeviceInitializer(adb_port=adb_port, device_serial=device_serial)

    # 设置root权限（刚启动的设备，需进行root权限设置）
    device_init.set_root_permission()
    logger.info(f"Device {device_serial} is now in root mode")

    apk_dir_extra = resolve_path(config.data_collect.apk_dir_extra)
    with resolve_path(config.data_collect.package_path_extra).open() as f:
        additional_packages = json.load(f)
    if package_name in additional_packages:
        apk_path = os.path.join(apk_dir_extra, package_name + ".apk")
        device_init.install_app(apk_path)
        time.sleep(4.0)
    logger.info(f"App {package_name} is now installed on device {device_serial}")

    # 授权
    device_init.grant_runtime_permissions(package_name=package_name)
    
    # 打开应用，确保有文件可以被拉取
    logger.info(f"Launching app {package_name} on device {device_serial}...")
    device = Device(device_serial)
    device.launch_app(package_name, front=True)
    time.sleep(5.0)
    device.wait_to_stabilize()
    device.stop_all_apps()
    time.sleep(3.0)

    # 保存初始快照
    device_init.save_initial_snapshot(package_name=package_name)

    # 关闭顶部显示点击位置的bar
    device_init.disable_pointer_location()

    # 系统时间设置
    device_init.set_system_time()


def run_gui_task(device, package, data_root_dir, log_file=None):
    """
    封装后的 GUI 初始化流程，可被其他模块直接调用。
    支持将所有输出写入 log_file。
    """
    # 如果需要写日志，则创建重定向上下文
    log_fp = None
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_fp = open(log_file, "w", encoding="utf-8")
        stdout_ctx = redirect_stdout(log_fp)
        stderr_ctx = redirect_stderr(log_fp)
    else:
        stdout_ctx = nullcontext()
        stderr_ctx = nullcontext()

    try:
        with stdout_ctx, stderr_ctx:
            device_init(
                device_serial=device,
                package_name=package
            )

            gui_init(
                device_serial=device,
                package_name=package,
                max_iterations=config.data_collect.max_iteration,
                max_tasks=config.data_collect.max_tasks,
                max_breadth=config.data_collect.max_breadth,
                max_depth=config.data_collect.max_depth,
                max_step=config.data_collect.max_step,
                add_step=config.data_collect.add_step,
                data_root_dir=data_root_dir,
            )
    finally:
        if log_fp:
            log_fp.close()

    return 0
