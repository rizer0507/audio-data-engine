#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在本机和 SSH 服务器之间上传 / 下载文件或目录，并显示实时进度。

只连你有权访问的主机。密码不要写进仓库；优先运行时输入，或用环境变量。

安装
----
  pip install paramiko

一次性传输
----------
  # 上传（本地根 → 远端根，目录内容一一对应，不再套一层同名目录）
  python scripts/sftp_transfer.py put --host 192.168.1.10 --user lizi --local D:\\Data\\batch_A --remote /data2/data-cp/lizi/batch_A

  # 下载
  python scripts/sftp_transfer.py get --host 192.168.1.10 --user lizi --remote /data2/data-cp/lizi/batch_A --local D:\\Data\\batch_A

  # 先看远端目录
  python scripts/sftp_transfer.py ls --host 192.168.1.10 --user lizi --remote /data2

Mac 双击启动
--------------
  双击 scripts/sftp_transfer.command（不要双击 .sh，Finder 会用编辑器打开）。
  首次若无法打开，在终端执行：chmod +x scripts/sftp_transfer.command

交互式（连上后反复指定路径，不用每次重输密码）
----------------------------------------------
  python scripts/sftp_transfer.py shell --host 192.168.1.10 --user lizi
  sftp> put D:\\Data\\a.wav /data2/a.wav
  sftp> get /data2/out D:\\Data\\out
  sftp> quit

密码
----
  未提供时会提示输入（不回显，推荐）。
  也可：--password、--password-file，或环境变量 SFTP_PASSWORD。
  --password 会出现在命令行历史和进程列表里，仅适合临时脚本。

中断后可加 --resume：跳过两端大小相同的文件，其余继续传。
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import shlex
import socket
import stat
import sys
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path


DEFAULT_PORT = 22
ENV_HOST = "SFTP_HOST"
ENV_USER = "SFTP_USER"
ENV_PORT = "SFTP_PORT"
ENV_PASSWORD = "SFTP_PASSWORD"


@dataclass(frozen=True)
class Item:
    local: Path
    remote: str
    size: int
    rel: str


@dataclass
class Plan:
    items: list[Item]
    skipped: int = 0
    skipped_bytes: int = 0


@dataclass
class Outcome:
    ok: int = 0
    ok_bytes: int = 0
    skipped: int = 0
    skipped_bytes: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    elapsed: float = 0.0


class TransferError(Exception):
    """单文件失败，连接仍可用。"""


class ConnectionLost(Exception):
    """连接已断，不能继续。"""


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def human_size(n: float) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def short(text: str, limit: int = 60) -> str:
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1) :]


def path_excluded(rel: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    parts: list[str] = []
    for part in rel.split("/"):
        parts.append(part)
        current = "/".join(parts)
        for pattern in patterns:
            if fnmatch(current, pattern) or fnmatch(part, pattern):
                return True
    return False


def read_password_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text


def resolve_password(args: argparse.Namespace) -> str:
    if args.key and args.password is None and args.password_file is None and os.environ.get(ENV_PASSWORD) is None:
        if not sys.stdin.isatty():
            return ""
        try:
            return getpass.getpass("私钥口令（没有请直接回车）: ")
        except (EOFError, KeyboardInterrupt) as exc:
            raise SystemExit("\n已取消。") from exc
    if args.password is not None:
        print(
            "[警告] --password 会出现在命令历史和进程列表中，建议改为运行时输入。",
            file=sys.stderr,
        )
        return args.password
    if args.password_file is not None:
        return read_password_file(Path(args.password_file))
    env = os.environ.get(ENV_PASSWORD)
    if env is not None:
        return env
    if not sys.stdin.isatty():
        raise SystemExit(
            f"非交互终端无法输入密码。请设置 {ENV_PASSWORD}，或使用 --password-file。"
        )
    prompt = f"密码 [{args.user}@{args.host}]: "
    try:
        return getpass.getpass(prompt)
    except (EOFError, KeyboardInterrupt) as exc:
        raise SystemExit("\n未输入密码，已取消。") from exc


def fingerprint(key) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    import base64

    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def host_names(host: str, port: int) -> list[str]:
    if port == 22:
        return [host]
    return [f"[{host}]:{port}", host]


def known_hosts_path() -> Path:
    return Path.home() / ".ssh" / "known_hosts"


def check_host_key(host: str, port: int, key, *, trust_host: bool, strict: bool) -> None:
    import paramiko

    path = known_hosts_path()
    host_keys = paramiko.HostKeys()
    if path.exists():
        host_keys.load(str(path))

    names = host_names(host, port)
    if any(host_keys.check(name, key) for name in names):
        return

    fp = fingerprint(key)
    has_entry = any(host_keys.lookup(name) for name in names)
    if has_entry and not trust_host:
        raise SystemExit(
            "主机密钥与 ~/.ssh/known_hosts 不一致，已中止（可能是服务器重装，也可能是中间人）。\n"
            f"本次密钥: {fp}\n"
            "若确认服务器身份有变，请加 --trust-host 后重试。"
        )
    if strict and not trust_host:
        raise SystemExit(
            f"未知主机 {host}:{port}，已按 --strict 中止。\n"
            f"密钥: {fp}\n"
            "确认后请去掉 --strict，或加上 --trust-host。"
        )

    record_as = names[0]
    line = f"{record_as} {key.get_name()} {key.get_base64()}\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if has_entry:
            kept = []
            for old in existing.splitlines(keepends=True):
                if any(old.startswith(name + " ") for name in names):
                    continue
                kept.append(old)
            existing = "".join(kept)
            if existing and not existing.endswith("\n"):
                existing += "\n"
        payload = existing + line
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
        saved = str(path)
    except OSError as exc:
        saved = f"未能写入 known_hosts ({exc})"

    if has_entry:
        print(f"[警告] 已按 --trust-host 覆盖记录主机密钥 {fp}")
        print(f"       {saved}")
    else:
        print(f"首次连接，已记录主机密钥 {fp}")
        print(f"记录位置: {saved}")
        print("之后若密钥变化将拒绝连接。")


def connect(args: argparse.Namespace, password: str):
    try:
        import paramiko
    except ImportError as exc:
        raise SystemExit(
            "缺少 paramiko，无法走 SFTP。请先安装：\n  pip install paramiko"
        ) from exc

    key = None
    if args.key:
        key_path = Path(args.key).expanduser()
        if not key_path.is_file():
            raise SystemExit(f"私钥不存在: {key_path}")
        passphrase = password or None
        errors: list[str] = []
        loaders = []
        for name in ("Ed25519Key", "RSAKey", "ECDSAKey"):
            key_cls = getattr(paramiko, name, None)
            if key_cls is not None:
                loaders.append(key_cls.from_private_key_file)
        for loader in loaders:
            try:
                key = loader(str(key_path), password=passphrase)
                break
            except Exception as exc:
                errors.append(str(exc))
        if key is None:
            raise SystemExit("无法读取私钥。若私钥有口令，请在密码提示里输入口令。\n" + errors[-1])

    try:
        sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
    except socket.gaierror as exc:
        raise SystemExit(f"无法解析主机 {args.host}: {exc}") from exc
    except TimeoutError as exc:
        raise SystemExit(f"连接 {args.host}:{args.port} 超时。") from exc
    except OSError as exc:
        raise SystemExit(f"无法连接 {args.host}:{args.port}: {exc}") from exc

    transport = paramiko.Transport(sock)
    transport.banner_timeout = args.timeout
    if args.compress:
        transport.use_compression(True)
    try:
        transport.default_window_size = 2147483647
        transport.packetizer.REKEY_BYTES = 1 << 40
        transport.packetizer.REKEY_PACKETS = 1 << 40
    except Exception:
        pass

    try:
        transport.start_client(timeout=args.timeout)
        remote_key = transport.get_remote_server_key()
        check_host_key(
            args.host,
            args.port,
            remote_key,
            trust_host=args.trust_host,
            strict=args.strict,
        )
        if key is not None:
            transport.auth_publickey(args.user, key)
        else:
            transport.auth_password(args.user, password)
    except paramiko.AuthenticationException as exc:
        transport.close()
        raise SystemExit(f"认证失败：用户名或密码不正确（{args.user}@{args.host}）。") from exc
    except paramiko.SSHException as exc:
        transport.close()
        raise SystemExit(f"SSH 握手失败: {exc}") from exc
    except Exception:
        transport.close()
        raise

    try:
        transport.set_keepalive(30)
        transport.sock.settimeout(None)
    except Exception:
        pass

    sftp = paramiko.SFTPClient.from_transport(transport)
    if sftp is None:
        transport.close()
        raise SystemExit("无法打开 SFTP 通道。")
    return transport, sftp


def close_session(sftp, transport) -> None:
    for obj in (sftp, transport):
        if obj is None:
            continue
        try:
            obj.close()
        except Exception:
            pass


def remote_error_missing(exc: BaseException) -> bool:
    errno = getattr(exc, "errno", None)
    if errno in (2, None) and "No such file" in str(exc):
        return True
    return errno == 2


def remote_kind(sftp, path: str) -> str:
    try:
        attr = sftp.stat(path)
    except OSError as exc:
        if remote_error_missing(exc):
            return "missing"
        raise
    mode = attr.st_mode or 0
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISLNK(mode):
        return "link"
    return "file"


def abs_remote(sftp, path: str) -> str:
    text = path.strip().replace("\\", "/")
    if not text:
        raise SystemExit("远端路径为空。")
    if text.startswith("/"):
        return text
    cwd = sftp.normalize(".")
    return f"{cwd.rstrip('/')}/{text}"


class RemoteDirs:
    def __init__(self, sftp) -> None:
        self.sftp = sftp
        self.known = {"/"}

    def ensure(self, directory: str) -> None:
        directory = directory.rstrip("/") or "/"
        if directory in self.known:
            return
        parent = directory.rsplit("/", 1)[0] or "/"
        if parent != directory:
            self.ensure(parent)
        kind = remote_kind(self.sftp, directory)
        if kind == "missing":
            self.sftp.mkdir(directory)
        elif kind != "dir":
            raise TransferError(f"远端路径已存在且不是目录: {directory}")
        self.known.add(directory)


def local_path(raw: str, *, must_exist: bool) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        path = path.resolve(strict=False)
    except OSError:
        path = path.absolute()
    if must_exist and not path.exists():
        raise SystemExit(f"本地路径不存在: {path}")
    return path


def dest_for_file(sftp, dest: str, filename: str) -> str:
    if dest.endswith("/"):
        return f"{dest.rstrip('/')}/{filename}"
    kind = remote_kind(sftp, dest)
    if kind == "dir":
        return f"{dest.rstrip('/')}/{filename}"
    return dest


def collect_local(local: Path, remote: str, sftp, excludes: list[str]) -> list[Item]:
    if local.is_file():
        dest = dest_for_file(sftp, remote, local.name)
        return [Item(local=local, remote=dest, size=local.stat().st_size, rel=local.name)]
    if not local.is_dir():
        raise SystemExit(f"本地路径既不是文件也不是目录: {local}")

    root = remote.rstrip("/") or "/"
    items: list[Item] = []
    for path in local.rglob("*"):
        rel = path.relative_to(local).as_posix()
        if path_excluded(rel, excludes):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        dest = f"/{rel}" if root == "/" else f"{root}/{rel}"
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise SystemExit(f"无法读取本地文件: {path} ({exc})") from exc
        items.append(Item(local=path, remote=dest, size=size, rel=rel))
    items.sort(key=lambda item: item.rel)
    return items


def walk_remote(sftp, root: str, excludes: list[str]) -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    stack = [(root.rstrip("/") or "/", "")]
    while stack:
        current, rel = stack.pop()
        try:
            entries = sftp.listdir_attr(current)
        except OSError as exc:
            raise SystemExit(f"无法列出远端目录 {current}: {exc}") from exc
        for attr in entries:
            name = attr.filename
            if name in (".", ".."):
                continue
            child_rel = name if not rel else f"{rel}/{name}"
            if path_excluded(child_rel, excludes):
                continue
            child = f"/{name}" if current == "/" else f"{current}/{name}"
            mode = attr.st_mode
            if not mode:
                try:
                    mode = sftp.stat(child).st_mode or 0
                except OSError:
                    continue
            if stat.S_ISLNK(mode):
                continue
            if stat.S_ISDIR(mode):
                stack.append((child, child_rel))
                continue
            if stat.S_ISREG(mode):
                found.append((child, int(attr.st_size or 0), child_rel))
    found.sort(key=lambda row: row[2])
    return found


def local_dest_for_file(dest: Path, filename: str) -> Path:
    if str(dest).endswith(("/", "\\")) or dest.is_dir():
        return dest / filename
    return dest


def collect_remote(remote: str, local: Path, sftp, excludes: list[str]) -> list[Item]:
    kind = remote_kind(sftp, remote)
    if kind == "missing":
        raise SystemExit(f"远端路径不存在: {remote}")
    if kind == "link":
        raise SystemExit(f"不跟随远端符号链接: {remote}")
    if kind == "file":
        attr = sftp.stat(remote)
        filename = remote.rstrip("/").rsplit("/", 1)[-1]
        dest = local_dest_for_file(local, filename)
        return [Item(local=dest, remote=remote, size=int(attr.st_size or 0), rel=filename)]

    rows = walk_remote(sftp, remote, excludes)
    items: list[Item] = []
    for remote_path, size, rel in rows:
        dest = local.joinpath(*rel.split("/"))
        items.append(Item(local=dest, remote=remote_path, size=size, rel=rel))
    return items


def apply_resume(sftp, items: list[Item], direction: str) -> Plan:
    kept: list[Item] = []
    skipped = 0
    skipped_bytes = 0
    print(f"比对已有文件（{len(items)} 个）…")
    for index, item in enumerate(items, start=1):
        if index == 1 or index % 200 == 0 or index == len(items):
            print(f"\r  {index}/{len(items)}", end="", flush=True)
        same = False
        try:
            if direction == "put":
                kind = remote_kind(sftp, item.remote)
                if kind == "file" and sftp.stat(item.remote).st_size == item.size:
                    same = True
            else:
                if item.local.is_file() and item.local.stat().st_size == item.size:
                    same = True
        except OSError:
            same = False
        if same:
            skipped += 1
            skipped_bytes += item.size
        else:
            kept.append(item)
    print()
    return Plan(items=kept, skipped=skipped, skipped_bytes=skipped_bytes)


class ByteSink:
    def __init__(self, progress, overall_id: int, file_id: int) -> None:
        self.progress = progress
        self.overall_id = overall_id
        self.file_id = file_id
        self.seen = 0

    def __call__(self, transferred: int, _total: int) -> None:
        delta = transferred - self.seen
        if delta < 0:
            delta = transferred
        self.seen = transferred
        if delta:
            self.progress.update(self.overall_id, advance=delta)
        self.progress.update(self.file_id, completed=transferred)

    def finish(self, size: int) -> None:
        remain = size - self.seen
        if remain > 0:
            self.progress.update(self.overall_id, advance=remain)
            self.seen = size
        self.progress.update(self.file_id, completed=size)


class _PlainProgress:
    """rich 不可用时的单行进度。接口只覆盖传输循环用到的方法。"""

    def __init__(self, total_bytes: int, file_count: int) -> None:
        self.total_bytes = total_bytes
        self.file_count = file_count
        self.done_bytes = 0
        self.started = time.perf_counter()
        self._last_draw = 0.0
        self._label = ""
        self.console = self

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        print()

    def add_task(self, description: str, total: int = 0) -> int:
        return 0

    def update(self, _task: int, *, advance: int = 0, completed: int | None = None, description: str = "") -> None:
        if advance:
            self.done_bytes += advance
        if description:
            self._label = description
        now = time.perf_counter()
        if now - self._last_draw < 0.1:
            return
        self._draw()
        self._last_draw = now

    def reset(self, _task: int, *, total: int, completed: int, description: str) -> None:
        self._label = description
        self._draw()

    def _draw(self) -> None:
        elapsed = max(time.perf_counter() - self.started, 1e-6)
        speed = self.done_bytes / elapsed
        label = short(self._label)
        if self.total_bytes > 0:
            pct = min(100.0, self.done_bytes / self.total_bytes * 100)
            remain = (self.total_bytes - self.done_bytes) / speed if speed > 0 else 0
            line = (
                f"{pct:5.1f}%  {human_size(self.done_bytes)}/{human_size(self.total_bytes)}  "
                f"{human_size(speed)}/s  剩余 {format_duration(remain)}  {label}"
            )
        else:
            line = label
        print("\r" + line[:120].ljust(120), end="", flush=True)

    def print(self, message: str) -> None:
        text = message.replace("[red]", "").replace("[/]", "")
        print("\n" + text)


def _open_progress(total_bytes: int, file_count: int):
    try:
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )
    except ImportError:
        return _PlainProgress(total_bytes, file_count)
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=28),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
        refresh_per_second=8,
    )


def _one_file(sftp, item: Item, direction: str, *, confirm: bool, dirs: RemoteDirs, callback) -> None:
    if direction == "put":
        parent = item.remote.rsplit("/", 1)[0] or "/"
        dirs.ensure(parent)
        sftp.put(str(item.local), item.remote, callback=callback, confirm=confirm)
        return
    item.local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(item.remote, str(item.local), callback=callback)


def transfer_items(
    sftp,
    items: list[Item],
    direction: str,
    *,
    confirm: bool,
) -> Outcome:
    import paramiko

    outcome = Outcome()
    dirs = RemoteDirs(sftp)
    started = time.perf_counter()
    total_bytes = sum(item.size for item in items)
    progress = _open_progress(total_bytes, len(items))
    lost = (paramiko.SSHException, EOFError, ConnectionError, TimeoutError, socket.error)
    try:
        with progress:
            overall_id = progress.add_task("总体", total=max(total_bytes, 1))
            file_id = progress.add_task("当前", total=1)
            for index, item in enumerate(items, start=1):
                label = short(item.rel)
                progress.update(overall_id, description=f"总体 {index - 1}/{len(items)}")
                progress.reset(file_id, total=max(item.size, 1), completed=0, description=label)
                sink = ByteSink(progress, overall_id, file_id)
                try:
                    _one_file(sftp, item, direction, confirm=confirm, dirs=dirs, callback=sink)
                    sink.finish(item.size)
                    outcome.ok += 1
                    outcome.ok_bytes += item.size
                except lost as exc:
                    outcome.failed.append((item.rel, str(exc)))
                    progress.console.print(f"[red]连接中断[/] {item.rel}: {exc}")
                    raise ConnectionLost(str(exc)) from exc
                except Exception as exc:
                    if _connection_dead(sftp):
                        outcome.failed.append((item.rel, str(exc)))
                        progress.console.print(f"[red]连接中断[/] {item.rel}: {exc}")
                        raise ConnectionLost(str(exc)) from exc
                    outcome.failed.append((item.rel, str(exc)))
                    progress.console.print(f"[red]失败[/] {item.rel}: {exc}")
            progress.update(overall_id, description=f"总体 {len(items)}/{len(items)}")
    finally:
        outcome.elapsed = time.perf_counter() - started
    return outcome


def _connection_dead(sftp) -> bool:
    try:
        channel = sftp.get_channel()
    except Exception:
        return True
    if channel is None:
        return True
    transport = getattr(channel, "transport", None)
    if transport is None:
        return False
    try:
        return not transport.is_active()
    except Exception:
        return True


def print_plan(direction: str, local: Path, remote: str, user: str, host: str, port: int, items: list[Item]) -> None:
    verb = "上传" if direction == "put" else "下载"
    total = sum(item.size for item in items)
    endpoint = f"{user}@{host}" if port == DEFAULT_PORT else f"{user}@{host}:{port}"
    print(f"{verb}  {len(items)} 个文件  {human_size(total)}")
    if direction == "put":
        print(f"  本地  {local}")
        print(f"  远端  {endpoint}:{remote}")
    else:
        print(f"  远端  {endpoint}:{remote}")
        print(f"  本地  {local}")
    if items:
        print(f"  示例  {items[0].rel}")
        if len(items) > 1:
            print(f"        {items[-1].rel}")
    print("目录按指定根路径对应，不会再套一层同名目录。")


def print_dry_run(items: list[Item], limit: int = 20) -> None:
    print("演练模式，不会传输。")
    for item in items[:limit]:
        print(f"  {item.rel}  {human_size(item.size)}")
    extra = len(items) - limit
    if extra > 0:
        print(f"  … 还有 {extra} 个")


def print_outcome(outcome: Outcome, *, resume_skipped: int, resume_bytes: int) -> None:
    skipped = outcome.skipped + resume_skipped
    skipped_bytes = outcome.skipped_bytes + resume_bytes
    speed = outcome.ok_bytes / outcome.elapsed if outcome.elapsed > 0 else 0.0
    print("完成")
    print(f"  成功  {outcome.ok} 个  {human_size(outcome.ok_bytes)}")
    print(f"  跳过  {skipped} 个  {human_size(skipped_bytes)}")
    print(f"  失败  {len(outcome.failed)} 个")
    print(f"  用时  {format_duration(outcome.elapsed)}  平均 {human_size(speed)}/s")
    if outcome.failed:
        print("失败明细:")
        for rel, reason in outcome.failed[:30]:
            print(f"  {rel}: {reason}")
        extra = len(outcome.failed) - 30
        if extra > 0:
            print(f"  … 还有 {extra} 个")


def run_transfer(
    sftp,
    args: argparse.Namespace,
    *,
    direction: str,
    local_raw: str,
    remote_raw: str,
) -> int:
    local = local_path(local_raw, must_exist=(direction == "put"))
    remote = abs_remote(sftp, remote_raw)
    excludes = list(args.exclude or [])

    if direction == "put":
        if local.is_dir() and remote_kind(sftp, remote) == "file":
            raise SystemExit(f"本地是目录，但远端已是文件: {remote}")
        print(f"正在扫描本地: {local}")
        items = collect_local(local, remote, sftp, excludes)
    else:
        if remote_kind(sftp, remote) == "dir" and local.exists() and not local.is_dir():
            raise SystemExit(f"远端是目录，但本地已是文件: {local}")
        print(f"正在扫描远端: {remote}")
        items = collect_remote(remote, local, sftp, excludes)

    host = args.host
    print_plan(direction, local, remote, args.user, host, args.port, items)
    if not items:
        print("没有需要传输的文件。")
        return 0
    if args.dry_run:
        print_dry_run(items)
        return 0

    plan = Plan(items=items)
    if args.resume:
        plan = apply_resume(sftp, items, direction)
        print(f"跳过大小相同的文件 {plan.skipped} 个（{human_size(plan.skipped_bytes)}）")
        if not plan.items:
            print("全部已存在且大小相同，无需传输。")
            return 0

    if direction == "get":
        for item in plan.items:
            item.local.parent.mkdir(parents=True, exist_ok=True)

    try:
        outcome = transfer_items(sftp, plan.items, direction, confirm=not args.no_confirm)
    except ConnectionLost as exc:
        print(f"\n连接中断: {exc}")
        print("可加 --resume 跳过大小相同的文件后继续。")
        return 2

    outcome.skipped = 0
    print_outcome(outcome, resume_skipped=plan.skipped, resume_bytes=plan.skipped_bytes)
    if outcome.failed:
        print("部分失败。可加 --resume 后重试未完成的文件（大小相同的会跳过）。")
        return 2
    return 0


def cmd_ls(sftp, remote_raw: str) -> int:
    remote = abs_remote(sftp, remote_raw)
    kind = remote_kind(sftp, remote)
    if kind == "missing":
        raise SystemExit(f"远端路径不存在: {remote}")
    if kind != "dir":
        attr = sftp.stat(remote)
        print(f"{human_size(int(attr.st_size or 0)):>12}  {remote}")
        return 0
    entries = sftp.listdir_attr(remote)
    entries.sort(key=lambda attr: attr.filename)
    print(remote)
    for attr in entries:
        name = attr.filename
        mode = attr.st_mode or 0
        if stat.S_ISDIR(mode):
            print(f"{'DIR':>12}  {name}/")
        else:
            print(f"{human_size(int(attr.st_size or 0)):>12}  {name}")
    return 0


def split_shell(line: str) -> list[str]:
    # Windows 路径里的反斜杠不能按 POSIX 转义处理。
    posix = os.name != "nt"
    parts = shlex.split(line, posix=posix)
    cleaned = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {'"', "'"}:
            part = part[1:-1]
        cleaned.append(part)
    return cleaned


def run_shell(sftp, args: argparse.Namespace) -> int:
    print(f"已连接 {args.user}@{args.host}:{args.port}")
    print("命令: put <本地> <远端> | get <远端> <本地> | ls [远端] | help | quit")
    last_code = 0
    while True:
        try:
            line = input("sftp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return last_code
        if not line:
            continue
        try:
            parts = split_shell(line)
        except ValueError as exc:
            print(f"无法解析命令: {exc}")
            continue
        cmd = parts[0].lower()
        if cmd in {"quit", "exit", "q"}:
            return last_code
        if cmd in {"help", "?"}:
            print("put <本地路径> <远端路径>    上传文件或目录")
            print("get <远端路径> <本地路径>    下载文件或目录")
            print("ls [远端路径]               查看远端目录")
            print("quit                        断开并退出")
            continue
        if cmd == "ls":
            target = parts[1] if len(parts) > 1 else "."
            try:
                last_code = cmd_ls(sftp, target)
            except SystemExit as exc:
                print(exc)
                last_code = 1
            continue
        if cmd in {"put", "get"}:
            if len(parts) != 3:
                print(f"用法: {cmd} <源路径> <目标路径>")
                last_code = 1
                continue
            src, dest = parts[1], parts[2]
            try:
                if cmd == "put":
                    last_code = run_transfer(
                        sftp, args, direction="put", local_raw=src, remote_raw=dest
                    )
                else:
                    last_code = run_transfer(
                        sftp, args, direction="get", local_raw=dest, remote_raw=src
                    )
            except SystemExit as exc:
                print(exc)
                last_code = 1
            except KeyboardInterrupt:
                print("\n本次传输已中断，连接仍保留。可加 --resume 在下次继续。")
                last_code = 130
            continue
        print(f"未知命令: {cmd}（输入 help）")


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    env_port = os.environ.get(ENV_PORT)
    parser.add_argument("--host", default=os.environ.get(ENV_HOST), help=f"服务器 IP 或主机名，也可用 {ENV_HOST}")
    parser.add_argument("--user", default=os.environ.get(ENV_USER), help=f"SSH 用户名，也可用 {ENV_USER}")
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_port) if env_port else DEFAULT_PORT,
        help=f"SSH 端口，默认 22，也可用 {ENV_PORT}",
    )
    parser.add_argument("--password", default=None, help="密码。不建议：会进入命令历史")
    parser.add_argument("--password-file", default=None, help="从文件读取密码（只用第一行，去掉末尾换行）")
    parser.add_argument("--key", default=None, help="私钥路径；提供后优先用密钥登录")
    parser.add_argument("--timeout", type=float, default=20, help="连接超时秒数，默认 20")
    parser.add_argument("--trust-host", action="store_true", help="主机密钥不一致时仍继续，并覆盖本地记录")
    parser.add_argument("--strict", action="store_true", help="未知主机也拒绝连接（默认首次连接会记录密钥后继续）")
    parser.add_argument("--compress", action="store_true", help="开启 SSH 压缩。音频等已压缩文件通常更慢，默认关闭")


def add_transfer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local", required=True, help="本地文件或目录的绝对或相对路径")
    parser.add_argument("--remote", required=True, help="远端路径。相对路径相对 SSH 登录后的家目录")
    parser.add_argument("--exclude", action="append", default=[], help="跳过匹配的相对路径或文件名，可重复。如 --exclude *.tmp")
    parser.add_argument("--resume", action="store_true", help="跳过两端大小相同的文件")
    parser.add_argument("--dry-run", action="store_true", help="只扫描并列出将传输的文件，不实际传输")
    parser.add_argument("--no-confirm", action="store_true", help="上传后不做大小确认，略快，默认会确认")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 SFTP 在本机和服务器之间传输文件或目录，并显示实时进度。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/sftp_transfer.py put --host 10.0.0.8 --user lizi "
            "--local D:\\Data\\batch_A --remote /data2/batch_A\n"
            "  python scripts/sftp_transfer.py get --host 10.0.0.8 --user lizi "
            "--remote /data2/batch_A --local D:\\Data\\batch_A\n"
            "  python scripts/sftp_transfer.py shell --host 10.0.0.8 --user lizi\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    put = sub.add_parser("put", help="上传：本地 → 服务器", parents=[])
    add_connection_args(put)
    add_transfer_args(put)

    get = sub.add_parser("get", help="下载：服务器 → 本地")
    add_connection_args(get)
    add_transfer_args(get)

    ls = sub.add_parser("ls", help="列出远端目录，用来确认路径")
    add_connection_args(ls)
    ls.add_argument("--remote", default=".", help="远端目录，默认登录后的当前目录")

    shell = sub.add_parser("shell", help="连上后反复 put/get/ls")
    add_connection_args(shell)
    shell.add_argument("--exclude", action="append", default=[], help="跳过匹配的相对路径或文件名，可重复")
    shell.add_argument("--resume", action="store_true", help="跳过两端大小相同的文件")
    shell.add_argument("--dry-run", action="store_true", help="只列出将传输的文件")
    shell.add_argument("--no-confirm", action="store_true", help="上传后不做大小确认")

    return parser


def require_connection(args: argparse.Namespace) -> None:
    missing = []
    if not args.host:
        missing.append("--host 或环境变量 SFTP_HOST")
    if not args.user:
        missing.append("--user 或环境变量 SFTP_USER")
    if missing:
        raise SystemExit("缺少连接参数: " + "，".join(missing))
    if not 1 <= args.port <= 65535:
        raise SystemExit(f"端口无效: {args.port}")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    require_connection(args)
    password = resolve_password(args)
    transport = None
    sftp = None
    try:
        print(f"正在连接 {args.user}@{args.host}:{args.port} …")
        transport, sftp = connect(args, password)
        print("已连接。")
        if args.cmd == "ls":
            return cmd_ls(sftp, args.remote)
        if args.cmd == "shell":
            return run_shell(sftp, args)
        return run_transfer(
            sftp,
            args,
            direction=args.cmd,
            local_raw=args.local,
            remote_raw=args.remote,
        )
    except KeyboardInterrupt:
        print("\n已中断。可加 --resume 跳过大小相同的文件后继续。")
        return 130
    finally:
        close_session(sftp, transport)


if __name__ == "__main__":
    sys.exit(main())
