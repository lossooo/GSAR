"""
ADB command utilities.
"""
import subprocess
import os
import time
import shlex
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


EXEC_LIMIT = 16
PUSH_PULL_LIMIT = 4

exec_semaphore = threading.Semaphore(EXEC_LIMIT)
push_and_pull_semaphore = threading.Semaphore(PUSH_PULL_LIMIT)

def run_adb_with_semaphore(full_cmd: list[str], type, timeout: int = 300):
    if type == 'adb_exec':
        with exec_semaphore:
            result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
            return result.stdout.strip(), result.stderr.strip()
    elif type in ['adb_pull', 'adb_push']:
        with push_and_pull_semaphore:
            subprocess.run(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True) #, timeout=pull_timeout)
            return True


def adb_exec(adb_port: int, device_id: str, cmd: str, use_shell: bool=True) -> str:
    """
    Execute an adb shell command and return output.

    :param adb_port: ADB server port.
    :param device_id: Target device/emulator ID.
    :param cmd: Shell command to execute.
    :param use_shell: Whether to use shell mode.
    :return: Command output as string.
    """
    if use_shell:
        full_cmd = ["adb", "-P", str(adb_port), "-s", device_id, "shell"] + cmd.split()
    else:
        full_cmd = ["adb", "-P", str(adb_port), "-s", device_id] + cmd.split()
    return run_adb_with_semaphore(full_cmd, "adb_exec")


def adb_pull(adb_port: int, device_id: str, remote_path: str, local_path: str) -> None:
    """
    Pull file from device to local path.

    :param adb_port: ADB server port.
    :param device_id: Target device/emulator ID.
    :param remote_path: Remote file path on device.
    :param local_path: Local file path to save.
    :return: None
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    full_cmd = ["adb", "-P", str(adb_port), "-s", device_id, "pull", remote_path, local_path]
    run_adb_with_semaphore(full_cmd, "adb_pull")


def adb_push(adb_port: int, device_id: str, local_path: str, remote_path: str) -> None:
    """
    Push file from local path to device.

    :param adb_port: ADB server port.
    :param device_id: Target device/emulator ID.
    :param local_path: Local file path to push.
    :param remote_path: Remote file path on device.
    :return: None
    """
    full_cmd = ["adb", "-P", str(adb_port), "-s", device_id, "push", local_path, remote_path]
    run_adb_with_semaphore(full_cmd, "adb_push")



def cmd_exec(cmd: str, timeout: int = 30) -> str:
    """
    运行本地命令，成功返回 stdout 文本；失败/超时抛 RuntimeError

    :return: Command output as string.
    """
    full_cmd = shlex.split(cmd)          # 安全拆成列表
    try:
        p = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"命令超时 ({timeout}s)") from e

    if p.returncode == 0:
        return p.stdout.strip()
    raise RuntimeError(f"命令失败 ({p.returncode}): {p.stderr}")



@dataclass
class DeviceInitializer:
    """设备初始化器"""
    adb_port: int          # ADB 端口
    device_serial: str     # 设备序列号

    def set_root_permission(self):
        """设置 root 权限"""
        adb_exec(self.adb_port, self.device_serial, "root", use_shell=False)
        time.sleep(5.0)

    def save_initial_snapshot(self, package_name: str):
        """保存初始快照"""
        adb_exec(self.adb_port, self.device_serial, f"mkdir -p /data/data/android_world/snapshots/{package_name}")
        time.sleep(2.0)
        adb_exec(self.adb_port, self.device_serial, f"cp -a /data/data/{package_name}/. /data/data/android_world/snapshots/{package_name}/")
        time.sleep(5.0)

    def disable_pointer_location(self):
        """关闭点击位置显示"""
        adb_exec(self.adb_port, self.device_serial, "settings put system pointer_location 0")

    def set_system_time(self):
        """设置系统时间"""
        adb_exec(self.adb_port, self.device_serial, "date -s '2023-01-01 00:00:00'")
        time.sleep(1.0)
        adb_exec(self.adb_port, self.device_serial, "settings put global auto_time 0")
        time.sleep(1.0)
        adb_exec(self.adb_port, self.device_serial, "settings put global auto_time_zone 0")
        time.sleep(1.0)
        adb_exec(self.adb_port, self.device_serial, "settings put system time_12_24 24")
        time.sleep(1.0)
        adb_exec(self.adb_port, self.device_serial, "service call alarm 3 s16 UTC")
        time.sleep(1.0)
        adb_exec(self.adb_port, self.device_serial, "date 111115002025")
        time.sleep(3.0)

    def stop_app(self, package_name: str):
        """停止应用运行"""
        adb_exec(self.adb_port, self.device_serial, f"am force-stop {package_name}")

    def clear_app_data(self, package_name: str):
        """清空应用数据"""
        # adb_exec(self.adb_port, self.device_serial, f"pm clear {package_name}")
        adb_exec(self.adb_port, self.device_serial, f"rm -rf /data/data/{package_name}/")

    def install_app(self, apk_path: str):
        """安装应用"""
        stdout, stderr = adb_exec(self.adb_port, self.device_serial, f"install -r {apk_path}", False)
        if "Success" not in stdout:
            raise RuntimeError(f"安装失败: {stderr}")

    def uninstall_app(self, package_name: str):
        """卸载应用"""
        stdout, stderr = adb_exec(self.adb_port, self.device_serial, f"uninstall {package_name}", False)
        if "Success" not in stdout:
            raise RuntimeError(f"卸载失败: {stderr}")
        
    def grant_runtime_permissions(self, package_name: str):
        """仅授予 APK 的 runtime 权限，跳过已授予和系统保护权限"""
        try:
            cmd = f"adb -P {str(self.adb_port)} -s {self.device_serial} shell dumpsys package {package_name} | grep permission"
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=True,  # 允许管道
                check=True
            )
        except subprocess.CalledProcessError:
            print(f"获取权限列表失败: {package_name}")
            return []

        permissions_to_grant = set()
        stdout = result.stdout
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("android.permission") and "granted=true" not in line and "flags=" in line:
                # 需要授权
                perm_name = line.split(":")[0].strip()
                permissions_to_grant.add(perm_name)

        # 执行授权命令
        permissions_to_grant = list(permissions_to_grant)
        for perm in permissions_to_grant:
            cmd = ["adb", "-s", self.device_serial, "shell", "pm", "grant", package_name, perm]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"成功授予: {perm}")
            except subprocess.CalledProcessError as e:
                print(f"授予失败: {perm} -> {e.stderr.strip()}")                

    def restore_selinux_context(self, package_name: str):
        """恢复应用目录的 SELinux 安全上下文"""
        adb_exec(self.adb_port, self.device_serial, f"restorecon -RD /data/data/{package_name}")

    def set_app_directory_permissions(self, package_name: str):
        """递归设置应用目录为 777 权限（允许所有读写执行）"""
        adb_exec(self.adb_port, self.device_serial, f"chmod 777 -R /data/data/{package_name}")

    def set_file_permissions(self, remote_path: str):
        """设置文件权限"""
        adb_exec(self.adb_port, self.device_serial, f"chmod 777 {remote_path}")

    def restore_file_selinux_context(self, remote_path: str):
        """恢复文件的 SELinux 安全上下文"""
        adb_exec(self.adb_port, self.device_serial, f"restorecon -RD {remote_path}")
