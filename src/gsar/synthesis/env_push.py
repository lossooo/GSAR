"""
Initialize the environment for the specified device and app, including uploading files (if needed), databases, and shared preferences.
"""
import os
import json
import time
import logging
import tempfile
import shutil

from .utils.adb_command import (
    adb_push,
    DeviceInitializer
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class Env_Push():
    """Used to push the environment for the specified device and app."""
    def __init__(self, root_dir: str, adb_port: int, device_serial: str, package_name: str, env_id: str=''):
        """
        :param root_dir: root directory of the project
        :param adb_port: adb port number, usually 5037
        :param device_serial: device ID, obtained by adb devices
        :param package_name: target app package name
        :param env_id: environment ID, used to identify the environment
        """
        self.root_dir = root_dir
        self.adb_port = adb_port
        self.device_serial = device_serial
        self.package_name = package_name
        self.env_id = env_id
        self.env_base_dir = os.path.join(self.root_dir, self.package_name, self.env_id)

        self.env_db_dir = os.path.join(self.env_base_dir, "databases")
        self.save_map_path = os.path.join(self.env_base_dir, "save_map.json")

        self.save_global_dir = os.path.join(self.root_dir, self.package_name, "global_files")
        os.makedirs(self.save_global_dir, exist_ok=True)
        self.manifest_file = os.path.join(self.root_dir, self.package_name, "global_manifest.json")

        self.device_init = DeviceInitializer(adb_port=self.adb_port, device_serial=self.device_serial)
        self.logger = logging.getLogger(self.__class__.__name__)


    def push(self):
        """
        Push the environment to the device
        """
        self.init_mobile_status()
        self.push_files()
        self.logger.info("File push completed, sleeping for 3 seconds")
        time.sleep(3)
        
    
    def init_mobile_status(self) -> None:
        """
        Initialize app environment, upload files (if needed), databases, and shared preferences
        1. Database files are in /<package_name>/databases/ directory
        2. Shared preferences files are in /<package_name>/shared_prefs/ directory
        3. Requires root permission or the app is debuggable
        """
        if not os.listdir(self.env_base_dir):
            self.logger.info(f"本地目录 {self.env_base_dir} 为空，跳过上传")
            return
        # 停止应用
        self.device_init.stop_app(self.package_name)

        # 设置系统时间
        self.device_init.set_system_time()

        # 先清除已有数据
        self.device_init.clear_app_data(self.package_name)
        self.logger.info(f"Deleted existing env for {self.package_name}")

        for item in os.listdir(self.env_base_dir):
            if item in ["save_map.json"]:
                continue    # 跳过上传保存信息json文件和文件目录
            local_item = os.path.join(self.env_base_dir, item)
            remote_item = f"/data/data/{self.package_name}/{item}"
            adb_push(self.adb_port, self.device_serial, local_item, remote_item)
            # self.logger.info(f"Uploaded {local_item} to {remote_item}")
        # 恢复 SELinux 安全上下文, 递归开放文件系统权限
        self.device_init.restore_selinux_context(self.package_name)
        self.device_init.set_app_directory_permissions(self.package_name)
        self.logger.info(f"Restored SELinux context and set permissions for {self.package_name}")


    def push_files(self) -> None:
        """
        Push the locally merged global files to the device, restoring the original local filenames.
        """
        if not os.path.exists(self.save_map_path):
            self.logger.info("Mapping file does not exist, no files need to be pushed")
            return
        with open(self.save_map_path, "r", encoding="utf-8") as file:
            mapping_list = json.load(file)

        # Create a temporary directory in advance to store files with "restored filenames"
        with tempfile.TemporaryDirectory() as tmpdir:
            for item in mapping_list:
                local_name = item["local"]          # Target filename
                global_name = item["global"]        # Source filename

                src = os.path.join(self.save_global_dir, global_name)
                if not os.path.isfile(src):
                    self.logger.warning(f"Global file missing, skipping: {src}")
                    continue

                dst = os.path.join(tmpdir, local_name)
                shutil.copy2(src, dst)

                remote_path = item["remote_file_path"]
                try:
                    adb_push(self.adb_port, self.device_serial, str(dst), remote_path)
                    self.device_init.restore_file_selinux_context(remote_path=remote_path)
                    self.device_init.set_file_permissions(remote_path=remote_path)
                    self.logger.info(f"Pushed {local_name} -> {remote_path}")
                except Exception as e:
                    self.logger.error(f"Failed to push {local_name}: {e}")
