"""Hail initialization utilities.

``import hail`` is cheap and does **not** boot the JVM (this Hail version
creates the SparkContext on the first Hail *object* use, not at import), so the
module imports Hail plainly at module scope and every ``from gnomad_meb.hail
import hail as hl`` call site is unchanged.

The JVM/Spark boot happens in one guarded seam, :func:`hail_init`: it points OS
fd 1 at fd 2 for the whole boot window, so log4j's startup banner lands on
stderr / the application log and stdout stays a clean payload stream for
``--dry-run``/``--describe``. It is idempotent, so an explicit call from the CLI
entry points (after their config is settled) is the single boot trigger; the
fd guard covers the import-time boot and ``hail.init`` alike because Hail may
boot on the first Hail object constructed inside ``hail_init``'s window.
"""

import atexit
import json
import os
import re
import shutil
from functools import lru_cache
from itertools import chain
from multiprocessing import cpu_count
from pathlib import Path
from shlex import split
from subprocess import check_output
from tempfile import gettempdir, mkdtemp
from typing import NamedTuple
from uuid import uuid4
from traceback import format_exc

import hail
from psutil import virtual_memory

from gnomad_meb.config import get_config
from gnomad_meb.logging import get_logger
from gnomad_meb.utils.hail_progress import SparkProgressReporter
from gnomad_meb.utils.logging import (
    attach_hail_logging,
    intercept_jvm_output,
    write_log4j_properties,
)

#: Whether Hail's JVM/Spark backend has been booted this process.
INITIALIZED: bool = False


def is_initialized() -> bool:
    """True once :func:`hail_init` has booted the backend."""
    return INITIALIZED


def get_node_local_scratch_root(
    basedir: Path | str = gettempdir(),
    force_node_local: bool = True,
    *,
    prefix: str | None = None,
) -> str:
    """A job-scoped, node-local tempdir created with ``mkdtemp``.

    The prefix defaults to ``gnomad-meb.<job>.<task>.``; callers may pass an
    explicit ``prefix`` (e.g. a resolved ``run_id``) instead.
    """
    if prefix is None:
        job = os.getenv("SLURM_JOB_ID", str(os.getpid()))
        task = os.getenv("SLURM_ARRAY_TASK_ID", "0")
        prefix = f"gnomad-meb.{job}.{task}."

    # By default, redirect known HPC shared filesystems to /tmp:
    basedir = str(basedir)
    if force_node_local and (basedir.startswith("/lustre") or basedir.startswith("/ceph")):
        get_logger().warning(f"refusing to write node-local temp files to shared filesystems, using /tmp instead: basedir = {basedir}")
        basedir = "/tmp"
    tempdir = mkdtemp(prefix=prefix, dir=basedir)
    return tempdir


def _expand_slurm_count_list(value: str | None) -> list[int]:
    def _expand_token(token: str):
        if "(x" in token:
            value, repeat_raw = token.strip().split("(x")
            repeat = int(repeat_raw.strip(")"))
            for _ in range(repeat):
                yield int(value)
        else:
            yield int(token.strip())

    if not value:
        return []

    return list(chain.from_iterable(map(_expand_token, value.split(","))))


def _expand_slurm_nodelist(nodelist: str | None) -> list[str]:
    def _split_nodelist(value: str):
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for char in value:
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            if char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        parts.append("".join(current))
        return parts

    if not nodelist:
        return []

    nodes: list[str] = []
    for token in _split_nodelist(nodelist):
        token = token.strip()
        if not token:
            continue
        if "[" not in token:
            nodes.append(token)
            continue
        prefix, _, rest = token.partition("[")
        for group in rest.rstrip("]").split(","):
            group = group.strip()
            if "-" not in group:
                nodes.append(prefix + group)
                continue
            start, _, end = group.partition("-")
            width = len(start)
            nodes.extend(
                prefix + str(i).zfill(width) for i in range(int(start), int(end) + 1)
            )
    return nodes


@lru_cache
def _get_slurm_cpu_settings() -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Get the cpu allocation for each node in the current job."""
    nodelist = _expand_slurm_nodelist(os.getenv("SLURM_JOB_NODELIST"))
    cpus_per_node = dict(zip(
        nodelist,
        _expand_slurm_count_list(os.getenv("SLURM_JOB_CPUS_PER_NODE", None)),
        strict=True,
    ))
    tasks_per_node = dict(zip(
        nodelist,
        _expand_slurm_count_list(os.getenv("SLURM_TASKS_PER_NODE", None)),
        strict=True,
    ))
    return nodelist, cpus_per_node, tasks_per_node


@lru_cache
def get_cpu_count() -> int:
    """CPUs available to the current task, for sizing local Spark/off-heap."""
    logger = get_logger()

    slurm_job_id = os.getenv("SLURM_JOB_ID", None)
    if slurm_job_id is None:
        return cpu_count()

    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK", None)
    if cpus_per_task is not None and cpus_per_task.strip():
        return max(int(cpus_per_task), 1)

    nodelist, cpus_per_node, tasks_per_node = _get_slurm_cpu_settings()
    node_id = os.getenv("SLURM_NODEID", None)

    if node_id is None:
        logger.warning(f"unknown compute node: node_id = {node_id}")
        return cpu_count()

    node_id = int(node_id)
    if int(node_id) < 0 or int(node_id) >= len(nodelist):
        logger.warning(f"invalid node id: node_id = {node_id}, nodelist = {nodelist}")
        return cpu_count()

    node_name = nodelist[int(node_id)]
    if node_name not in cpus_per_node or node_name not in tasks_per_node:
        logger.warning(f"no resources found for node: node_name = {node_name}, cpus = {cpus_per_node}, tasks = {tasks_per_node}")
        return cpu_count()

    return max(cpus_per_node[node_name] // tasks_per_node[node_name], 1)


def get_cpu_count_slurm():
    slurm_job_id = os.getenv("SLURM_JOB_ID", None)
    if slurm_job_id is None:
        return cpu_count()

    _, cpus, _ = _get_slurm_cpu_settings()
    if cpus:
        return sum(cpus.values())
    else:
        return cpu_count()


@lru_cache
def get_memory(allocation_target: float = 0.95):
    logger = get_logger()
    default_memory = f"{round(virtual_memory().total * allocation_target) >> 30}G"
    slurm_job_id = os.getenv("SLURM_JOB_ID", None)
    if slurm_job_id is not None:
        try:
            logger.debug("trying to get memory allocation from slurm job")
            stdout = check_output(
                split(f"scontrol -o show job {slurm_job_id}")
            ).decode()
            tres_match = re.search(
                r"\s(ReqTRES|AllocTRES|TRES)=(.*?)(mem=(?P<mem>[\d.kKmMgGtT]+),?)(.*?)\s",
                stdout,
            )
            if tres_match is not None:
                tres_mem = tres_match.group("mem")
                if tres_mem[-1] in "tT":
                    mem_bytes = int(float(tres_mem[:-1]) * (2**40))
                elif tres_mem[-1] in "gG":
                    mem_bytes = int(tres_mem[:-1]) << 30
                elif tres_mem[-1] in "mM":
                    mem_bytes = int(tres_mem[:-1]) << 20
                elif tres_mem[-1] in "kK":
                    mem_bytes = int(tres_mem[:-1]) << 10
                else:
                    logger.error(f"error converting tres memory value {tres_mem}")
                    return default_memory

                num_tasks_match = re.search(r"\sNumTasks=(?P<num_tasks>\d+)\s", stdout)
                num_tasks = int(num_tasks_match.group("num_tasks")) if num_tasks_match is not None else 1

                logger.debug(f"computing per worker memory: mem_bytes = {mem_bytes}, num_tasks = {num_tasks}")
                if num_tasks > 1:
                    mem_bytes //= num_tasks

                if allocation_target < 1.0:
                    mem_bytes = int(mem_bytes * allocation_target)
                
                return f"{mem_bytes >> 30}g"
            else:
                logger.warning("found no memory TRES allocation in scontrol output")
                return default_memory
        except Exception as e:
            logger.error(f"error getting memory allocation from slurm: {e}")
            return default_memory
    return default_memory


class HailMemorySettings(NamedTuple):
    node_memory_gb: int
    overhead_total_gb: int
    executor_memory_gb: int
    driver_memory_gb: int
    driver_max_result_size_mb: int

    @property
    def driver_memory(self):
        return f"{self.driver_memory_gb}G"

    @property
    def executor_memory(self):
        return f"{self.executor_memory_gb}G"

    @property
    def memory_overhead(self):
        return f"{self.overhead_total_gb}G"

    @property
    def driver_max_result_size(self):
        return f"{self.driver_max_result_size_mb}m"

    def get_off_heap_memory_per_core_mb(self, ncores: int = 1):
        return (int(self.overhead_total_gb) * 1_024) // ncores


class HailTempSettings(NamedTuple):
    """The temp locations Hail/Spark need, resolved coherently from settings.

    - ``spark_local_dir``: **node-local** scratch for Spark shuffle/spill,
      created with ``mkdtemp`` and prefixed with the run id. Kept node-local on
      purpose -- spill wants fast local disk.
    - ``hail_tmp``: Hail's staging/intermediate dir, on the **shared** run
      scratch, suffixed with ``settings.run_id`` so every task of one logical
      run shares it while distinct runs and array tasks never collide.
    - ``hail_local_dir``: Hail's per-process local dir, on the **shared** run
      scratch -- deliberately not node-local ``/tmp``, so it survives
      node-local temp pressure -- suffixed randomly so concurrent processes
      never collide.
    """

    spark_local_dir: str
    hail_tmp: str
    hail_local_dir: str

    def delete_temp_dirs(self, ignore_errors: bool = True):
        """Delete all temporary directories."""
        get_logger().debug(f"deleting temporary directories: tempdirs = {self}")
        try:
            shutil.rmtree(self.spark_local_dir)
            shutil.rmtree(self.hail_local_dir)
            shutil.rmtree(self.hail_tmp)
        except Exception as e:
            get_logger().error(f"failed to delete tmporary directories: tempdirs = {self}, error = {e}")
            get_logger().error(format_exc())
            if not ignore_errors:
                raise

    @classmethod
    def from_config(
        cls,
        *,
        run_id: str | None = None,
        shared_root: str | None = None,
        spark_basedir: Path | str = gettempdir(),
    ) -> "HailTempSettings":
        """Resolve all three temp locations from config and environment.

        ``run_id``, ``shared_root`` and ``spark_basedir`` are overridable so
        tests (and callers needing a fixed layout) can pin them.
        """
        cfg = get_config()
        rid = run_id or cfg.run_id

        # Spark's node-local scratch: prefer SCRATCH_DIR, else the node-local
        # default; get_node_local_scratch_root redirects shared filesystems to
        # /tmp and mkdtemp's a run-id-prefixed dir.
        spark_base = os.getenv("SCRATCH_DIR") or spark_basedir
        spark_local_dir = get_node_local_scratch_root(
            spark_base,
            force_node_local=True,
            prefix=f"{rid}.spark.",
        )

        # Shared run scratch: the operator's configured path, else a sibling of
        # the run scratch dir.
        if shared_root:
            root = Path(shared_root)
        elif cfg.hail.tmp_dir:
            root = Path(cfg.hail.tmp_dir)
        else:
            root = Path(cfg.scratch_dir) / "hail-tmp"

        hail_tmp = root / rid
        hail_local_dir = root / f"{rid}.local.{uuid4().hex[:8]}"

        return cls(
            spark_local_dir=spark_local_dir,
            hail_tmp=str(hail_tmp.resolve()),
            hail_local_dir=str(hail_local_dir.resolve()),
        )


def get_hail_memory_settings(allocation_target: float = 0.8):
    """Get hail memory settings to avoid Slurm OOM errors."""
    cfg = get_config()
    node_memory = get_memory(allocation_target=allocation_target).lower()
    node_memory_gb = int(node_memory[:-1])

    overhead_total_gb = int(cfg.hail.memory_overhead) // 1_024
    executor_memory_gb = max(node_memory_gb - overhead_total_gb - 4, 1)
    if cfg.hail.driver_memory_gb is not None:
        driver_memory_gb = int(cfg.hail.driver_memory_gb)
    else:
        driver_memory_gb = min(
            max(int(node_memory_gb * cfg.hail.driver_memory_fraction), 4),
            cfg.hail.driver_max_memory_gb,
        )

    return HailMemorySettings(
        node_memory_gb=node_memory_gb,
        overhead_total_gb=overhead_total_gb,
        executor_memory_gb=executor_memory_gb,
        driver_memory_gb=driver_memory_gb,
        driver_max_result_size_mb=cfg.hail.driver_max_result_size,
    )

def get_default_partitions(partitions_per_core: int | None = None) -> int:
    """Retrieve a reasonable default partition count for hail tables.

    This function infers the number of available CPU's for the hail runtime
    in the current environment from the configuration parameter n_cores,
    slurm resources, or the current machine CPU count and uses
    `partitions_per_core` as multiplier to generate the partition count.
    """
    cfg = get_config()
    if partitions_per_core is None:
        partitions_per_core = cfg.hail.partitions_per_core

    if cfg.n_cores is None:
        return get_cpu_count_slurm() * partitions_per_core
    else:
        return cfg.n_cores * partitions_per_core


class HailStructEncoder(json.JSONEncoder):
    """Safely convert a hail struct to a dictionary and json-encode it."""

    def default(self, o):
        if isinstance(o, hail.Struct):
            data_d = {}
            for field, value in o.items():
                if isinstance(value, hail.expr.Expression):
                    data_d[field] = self.default(hail.eval(value))
                else:
                    data_d[field] = value
            return data_d
        if isinstance(o, set):
            return list(o)
        return super().default(o)


def hail_init():
    """Boot Hail once; the fd-1 guard spans the entire boot window.

    Idempotent: the second and later calls are no-ops. fd 1 is pointed at fd 2
    for the whole body -- ``hail.init``, ``_set_flags``, the log bridge and the
    reporter start -- so the JVM/Spark banners land on stderr however Hail's
    lazy backend comes up. ``INITIALIZED`` is set before the body so a
    re-entrant boot is impossible; no lock is needed (single-threaded startup).
    """
    global INITIALIZED
    if INITIALIZED:
        return
    INITIALIZED = True

    cfg = get_config()
    logger = get_logger()

    # Write the JVM's log4j config here, not only in ``configure_logging``: the
    # config-free ``run <manifest>`` path loads its settings later, and playbooks
    # call ``hail_init`` directly, so every settings-loading entry point must get
    # log4j routed to stderr at ``Settings.hail.log_level``.
    log4j_properties = write_log4j_properties(
        cfg.scratch_dir, level=cfg.hail.log_level
    )

    if (
        cfg.checkpoint is not None
        and not os.path.isdir(cfg.checkpoint)
        and not str(cfg.checkpoint).startswith("s3")
    ):
        logger.debug(f"creating checkpoint dir at {cfg.checkpoint}")
        os.makedirs(cfg.checkpoint, exist_ok=True)

    master_ip = os.getenv("SPARK_MASTER_IP", None)
    master_port = os.getenv("SPARK_MASTER_PORT", None)

    memory_settings = get_hail_memory_settings()
    if master_ip is None or master_port is None:
        master_url = f"local[{get_cpu_count()}]"
        logger.info("no spark cluster found, starting local hail")
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            f"--master local[{get_cpu_count()}] --driver-memory {memory_settings.driver_memory} --packages {','.join(cfg.hail.additional_packages)} pyspark-shell"
        )
        logger.debug(f"using PYSPARK_SUBMIT_ARGS='{os.getenv('PYSPARK_SUBMIT_ARGS')}'")
    else:
        master_url = f"spark://{master_ip}:{master_port}"
        logger.info(f"starting hail on spark cluster '{master_url}'")
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            f'--master "{master_url}" --driver-memory {memory_settings.driver_memory} --packages {",".join(cfg.hail.additional_packages)} pyspark-shell'
        )
        # os.environ["PYSPARK_SUBMIT_ARGS"] = f"--master '{master_url}' --packages {','.join(cfg.hail.additional_packages)} pyspark-shell"

    tmp_settings = HailTempSettings.from_config()
    atexit.register(tmp_settings.delete_temp_dirs, ignore_errors=True)

    logger.debug(
        f"initializing hail: master = {master_url}, "
        f"tmp_settings = {tmp_settings}, "
        f"memory = {memory_settings}, cpu_count = {get_cpu_count()}"
    )

    with intercept_jvm_output():
        hail.init(
            master=master_url,
            idempotent=True,
            tmp_dir=tmp_settings.hail_tmp,
            local_tmpdir=tmp_settings.hail_local_dir,
            log=os.path.join(cfg.scratch_dir, "hail.log"),
            show_progress=False,  # Spark's own bar is replaced by the reporter
            quiet=True,  # the file bridge mirrors the log; no built-in stderr copy
            # copy_spark_log_on_error=True,
            spark_conf={
                "spark.local.dir": tmp_settings.spark_local_dir,
                "spark.speculation": "false",
                "spark.task.maxFailures": str(cfg.hail.max_failures),
                "spark.network.timeout": "600s",
                "spark.core.connection.ack.wait.timeout": "600s",
                "spark.rpc.message.maxSize": "512",
                "spark.driver.memory": memory_settings.driver_memory,
                "spark.driver.memoryOverhead": memory_settings.memory_overhead,
                "spark.driver.maxResultSize": memory_settings.driver_max_result_size,
                "spark.executor.memory": memory_settings.executor_memory,
                "spark.executor.memoryOverhead": str(cfg.hail.memory_overhead),
                "spark.executorEnv.HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB": str(int(memory_settings.get_off_heap_memory_per_core_mb(get_cpu_count()) * 0.9)),
                "spark.executorEnv.TMPDIR": tmp_settings.spark_local_dir,
                "spark.executorEnv.SPARK_MEM": memory_settings.executor_memory,
                "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
                "spark.hadoop.fs.s3a.endpoint": cfg.s3.endpoint_url,
                "spark.hadoop.fs.s3a.access.key": cfg.s3.access_key,
                "spark.hadoop.fs.s3.awsAccessKeyId": cfg.s3.access_key,
                "spark.hadoop.fs.s3a.secret.key": cfg.s3.secret_key,
                "spark.hadoop.fs.s3.awsSecretAccessKey": cfg.s3.secret_key,
                "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                "spark.hadoop.fs.s3.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                "spark.hadoop.fs.s3a.path.style.access": "true",
                "spark.hadoop.fs.s3a.create.storage.class": cfg.s3.storage_class,
                "spark.hadoop.fs.s3a.connection.establish.timeout": str(
                    cfg.hail.s3_timeout_ms
                ),
                "spark.hadoop.fs.s3a.connection.maximum": str(cfg.hail.s3_max_connections),
                "spark.hadoop.fs.s3a.connection.request.timeout": str(
                    cfg.hail.s3_request_timeout_ms
                ),
                "spark.hadoop.fs.s3a.connection.acquisition.timeout": str(
                    cfg.hail.s3_request_timeout_ms
                ),
                "spark.hadoop.fs.s3a.connection.idle.time": str(cfg.hail.s3_timeout_ms),
                "spark.hadoop.fs.s3.maxConnections": str(cfg.hail.s3_max_connections),
                "spark.driver.extraJavaOptions": f"-Dlog4j.configurationFile={log4j_properties} -Xss4M -Dcom.amazonaws.sdk.disableCertChecking={str(not cfg.s3.verify).lower()} -XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35",
                "spark.executor.extraJavaOptions": f"-Dlog4j.configurationFile={log4j_properties} -Xss4M -Dcom.amazonaws.sdk.disableCertChecking={str(not cfg.s3.verify).lower()} -XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35",
                "spark.ui.showConsoleProgress": "false",
            },
        )
        hail._set_flags(use_new_shuffle="1")

        attach_hail_logging(os.path.join(cfg.scratch_dir, "hail.log"))
        reporter = SparkProgressReporter().start()
        atexit.register(reporter.stop)

        logger.debug(
            f"effective spark configuration: conf = \n{hail.spark_context().getConf().toDebugString()}",
        )
