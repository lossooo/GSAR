"""Capture an app state so it can be restored in a later synthesis iteration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid

from ..common.model_client import MLLM_Model
from ..common.prompts import PROMPT_TEMPLATE_FIND_FILE_FIELDS
from .utils.adb_command import adb_exec, adb_pull
from .utils.sql_command import get_table_schema, list_tables, sample_rows_ignore_blob
from .utils.utils import extract_last_json

logger = logging.getLogger(__name__)


class Env_Pull:
    """Pull app data and preserve files referenced outside the app data directory."""

    def __init__(
        self,
        adb_port: int,
        device_serial: str,
        package_name: str,
        root_dir: str = ".",
    ):
        self.root_dir = root_dir
        self.adb_port = adb_port
        self.device_serial = device_serial
        self.package_name = package_name
        self.env_id = uuid.uuid4().hex
        self.env_base_dir = os.path.join(root_dir, package_name, self.env_id)
        self.env_db_dir = os.path.join(self.env_base_dir, "databases")
        self.save_map_path = os.path.join(self.env_base_dir, "save_map.json")
        self.staging_dir = os.path.join(self.env_base_dir, "captured_external_files")
        self.save_global_dir = os.path.join(root_dir, package_name, "global_files")
        self.manifest_file = os.path.join(root_dir, package_name, "global_manifest.json")

        os.makedirs(self.env_base_dir, exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.save_global_dir, exist_ok=True)

    def __del__(self):
        shutil.rmtree(self.staging_dir, ignore_errors=True)

    def pull(self) -> str:
        """Capture app data and return the generated environment ID."""
        self.get_current_status()
        if not os.path.isdir(self.env_db_dir):
            logger.info("No database directory found; skipping referenced-file capture")
            return self.env_id

        for filename in os.listdir(self.env_db_dir):
            database_path = os.path.join(self.env_db_dir, filename)
            if filename.endswith(("-wal", "-shm")):
                continue
            if filename.endswith((".db", ".sqlite")) or "." not in filename:
                self.save_data_file(database_path)

        if os.path.isfile(self.save_map_path):
            self.process_files()
        return self.env_id

    def get_current_status(self) -> None:
        """Pull the current contents of the app data directory."""
        remote_base = f"/data/data/{self.package_name}/"
        stdout, _ = adb_exec(
            self.adb_port,
            self.device_serial,
            f"ls -1 {remote_base}",
        )
        if not stdout:
            logger.info("App data directory not found for %s", self.package_name)
            return

        for subdirectory in stdout.splitlines():
            remote_path = f"{remote_base}/{subdirectory}"
            local_path = os.path.join(self.env_base_dir, subdirectory)
            adb_pull(self.adb_port, self.device_serial, remote_path, local_path)
            time.sleep(3)
        logger.info("Saved app state to %s", self.env_base_dir)

    def save_data_file(self, database_path: str) -> None:
        """Preserve external files whose paths are referenced by an app database."""
        database_description = ""
        with sqlite3.connect(database_path) as connection:
            for table in list_tables(connection):
                schema = get_table_schema(connection, table)
                rows = sample_rows_ignore_blob(connection, table)
                if rows:
                    database_description += (
                        f"DB: {os.path.basename(database_path)}, Table: {table}\n"
                        f"Schema: {schema}\nSample: {rows}\n\n"
                    )
        if not database_description:
            return

        model = MLLM_Model()
        response, _ = model.get_model_response(
            prompt=PROMPT_TEMPLATE_FIND_FILE_FIELDS.format(
                db_info=database_description
            )
        )
        references = extract_last_json(response)
        if not isinstance(references, list):
            logger.info("No external file fields found in %s", database_path)
            return

        if os.path.isfile(self.save_map_path):
            with open(self.save_map_path, encoding="utf-8") as handle:
                save_map = json.load(handle)
        else:
            save_map = []

        for reference in references:
            db_name = reference.get("database", "")
            table_name = reference.get("table", "")
            field_name = reference.get("field", "")
            suffix = reference.get("suffix", "")
            if not all((db_name, table_name, field_name, suffix)):
                continue

            referenced_database = os.path.join(self.env_db_dir, db_name)
            with sqlite3.connect(referenced_database) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(f"SELECT {field_name} FROM {table_name} LIMIT 15")
                    rows = cursor.fetchall()
                except Exception as exc:
                    logger.warning("Failed to read file references: %s", exc)
                    continue

            for row in rows:
                remote_path = row[0]
                if not isinstance(remote_path, str) or not remote_path.endswith(suffix):
                    continue
                stdout, _ = adb_exec(
                    self.adb_port,
                    self.device_serial,
                    f"if [ -f {remote_path} ]; then echo 1; else echo 0; fi",
                )
                if stdout.strip() != "1":
                    continue

                local_path = os.path.join(
                    self.staging_dir,
                    os.path.basename(remote_path),
                )
                adb_pull(
                    self.adb_port,
                    self.device_serial,
                    remote_path,
                    local_path,
                )
                save_map.append(
                    {
                        "remote_file_path": remote_path,
                        "local_file_path": local_path,
                    }
                )
                time.sleep(2)

        with open(self.save_map_path, "w", encoding="utf-8") as handle:
            json.dump(save_map, handle, ensure_ascii=False, indent=2)

    def process_files(self) -> None:
        """Deduplicate captured external files and finalize their restore map."""
        with open(self.save_map_path, encoding="utf-8") as handle:
            mapping = json.load(handle)
        if os.path.isfile(self.manifest_file):
            with open(self.manifest_file, encoding="utf-8") as handle:
                manifest = json.load(handle)
        else:
            manifest = {}

        hash_to_global = {
            details["hash"]: filename
            for filename, details in manifest.items()
        }
        finalized = []
        for item in mapping:
            local_path = item["local_file_path"]
            if not os.path.isfile(local_path):
                continue
            local_name = os.path.basename(local_path)
            digest = self._get_sha256(local_path)

            if digest in hash_to_global:
                global_name = hash_to_global[digest]
                names = set(manifest[global_name]["original_names"])
                manifest[global_name]["original_names"] = sorted(names | {local_name})
            else:
                global_name = f"{uuid.uuid4().hex}_{local_name}"
                shutil.copy2(
                    local_path,
                    os.path.join(self.save_global_dir, global_name),
                )
                manifest[global_name] = {
                    "hash": digest,
                    "original_names": [local_name],
                }
                hash_to_global[digest] = global_name

            finalized.append(
                {
                    "remote_file_path": item["remote_file_path"],
                    "local": local_name,
                    "global": global_name,
                }
            )

        with open(self.save_map_path, "w", encoding="utf-8") as handle:
            json.dump(finalized, handle, ensure_ascii=False, indent=2)
        with open(self.manifest_file, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        shutil.rmtree(self.staging_dir, ignore_errors=True)

    @staticmethod
    def _get_sha256(file_path: str) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            while chunk := handle.read(1 << 16):
                digest.update(chunk)
        return digest.hexdigest()
