# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Argument-level inspection capabilities for script runners and nested CLIs.

Unknown actions of these tools require execution/write permission. A command
family name alone never establishes that its nested operation is an inspection.
"""

from __future__ import annotations

import re

from victor.security.shell_syntax import ShellWord

# These programs evaluate caller/project code; benign installed-binary metadata
# flags are the only general exception. The explicit execution path remains the
# owner of tests, builds, plugins, source files, and arbitrary code.
OPAQUE_PROGRAMS = frozenset(
    {
        ".",
        "source",
        "awk",
        "python",
        "python3",
        "node",
        "pytest",
        "tox",
        "make",
        "mypy",
        "pylint",
        "flake8",
        "eslint",
        "terraform",
    }
)

_INSPECTION_PATHS: dict[str, frozenset[tuple[str, ...]]] = {
    "gh": frozenset(
        {
            ("pr", "view"),
            ("pr", "list"),
            ("pr", "diff"),
            ("pr", "checks"),
            ("pr", "status"),
            ("issue", "view"),
            ("issue", "list"),
            ("issue", "status"),
            ("repo", "view"),
            ("repo", "list"),
            ("release", "view"),
            ("release", "list"),
            ("workflow", "view"),
            ("workflow", "list"),
            ("run", "view"),
            ("run", "list"),
            ("run", "watch"),
            ("auth", "status"),
            ("config", "get"),
            ("config", "list"),
            ("secret", "list"),
            ("variable", "get"),
            ("variable", "list"),
            ("search",),
        }
    ),
    "docker": frozenset(
        {
            ("ps",),
            ("images",),
            ("inspect",),
            ("search",),
            ("info",),
            ("version",),
            ("stats",),
            ("history",),
            ("logs",),
            ("top",),
            ("port",),
            ("network", "ls"),
            ("network", "inspect"),
            ("volume", "ls"),
            ("volume", "inspect"),
            ("system", "df"),
            ("system", "info"),
            ("system", "events"),
            ("context", "ls"),
            ("context", "show"),
            ("context", "inspect"),
            ("buildx", "ls"),
            ("buildx", "inspect"),
            ("buildx", "version"),
            ("manifest", "inspect"),
            ("image", "ls"),
            ("image", "inspect"),
            ("image", "history"),
            ("container", "ls"),
            ("container", "inspect"),
            ("container", "stats"),
            ("compose", "ps"),
            ("compose", "logs"),
            ("compose", "config"),
            ("compose", "top"),
            ("compose", "images"),
            ("compose", "port"),
            ("compose", "version"),
            ("compose", "ls"),
        }
    ),
    "helm": frozenset(
        {
            ("list",),
            ("ls",),
            ("status",),
            ("history",),
            ("version",),
            ("repo", "list"),
            ("search", "repo"),
            ("search", "hub"),
            ("get", "all"),
            ("get", "values"),
            ("get", "manifest"),
            ("get", "notes"),
            ("get", "hooks"),
            ("get", "metadata"),
            ("show", "all"),
            ("show", "chart"),
            ("show", "values"),
            ("show", "readme"),
            ("show", "crds"),
        }
    ),
    "kubectl": frozenset(
        {
            ("get",),
            ("describe",),
            ("logs",),
            ("top",),
            ("api-resources",),
            ("api-versions",),
            ("cluster-info",),
            ("version",),
            ("explain",),
            ("auth", "can-i"),
            ("auth", "whoami"),
            ("config", "view"),
            ("config", "current-context"),
            ("config", "get-contexts"),
            ("config", "get-clusters"),
            ("config", "get-users"),
        }
    ),
    "az": frozenset(
        {
            ("account", "list"),
            ("account", "show"),
            ("config", "get"),
            ("group", "list"),
            ("group", "show"),
            ("resource", "list"),
            ("resource", "show"),
            ("monitor", "activity-log", "list"),
            ("monitor", "metrics", "list"),
            ("monitor", "metrics", "list-definitions"),
            ("monitor", "log-analytics", "query"),
        }
    ),
    "pip": frozenset(
        {
            ("list",),
            ("show",),
            ("freeze",),
            ("check",),
            ("debug",),
            ("help",),
            ("inspect",),
            ("config", "get"),
            ("config", "list"),
            ("config", "debug"),
            ("cache", "info"),
            ("cache", "list"),
            ("cache", "dir"),
            ("index", "versions"),
        }
    ),
    "npm": frozenset(
        {
            ("list",),
            ("ls",),
            ("view",),
            ("show",),
            ("info",),
            ("search",),
            ("help",),
            ("outdated",),
            ("explain",),
            ("query",),
            ("why",),
            ("config", "get"),
            ("config", "list"),
            ("pkg", "get"),
        }
    ),
    "yarn": frozenset({("list",), ("ls",), ("info",), ("why",), ("help",), ("cache", "list")}),
    "pnpm": frozenset(
        {("list",), ("ls",), ("view",), ("show",), ("info",), ("why",), ("help",), ("outdated",)}
    ),
    "cargo": frozenset({("tree",), ("metadata",), ("version",), ("help",)}),
    "go": frozenset({("version",), ("help",)}),
    "ip": frozenset(
        {
            ("addr", "show"),
            ("address", "show"),
            ("route", "show"),
            ("route", "list"),
            ("route", "get"),
            ("link", "show"),
            ("neigh", "show"),
            ("netns", "list"),
        }
    ),
}
_INSPECTION_PATHS["podman"] = _INSPECTION_PATHS["docker"]
_INSPECTION_PATHS["pip3"] = _INSPECTION_PATHS["pip"]

_GLOBAL_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "gh": frozenset({"-R", "--repo", "--hostname"}),
    "kubectl": frozenset(
        {"-n", "--namespace", "--context", "--cluster", "--user", "--server", "-s"}
    ),
    "helm": frozenset({"-n", "--namespace", "--kube-context"}),
    "docker": frozenset({"-H", "--host", "--context"}),
    "podman": frozenset({"--url", "--connection"}),
}


def _action_args(tool: str, args: list[str]) -> list[str]:
    """Skip only known inert global connection/namespace flags."""
    i = 0
    values = _GLOBAL_VALUE_FLAGS.get(tool, frozenset())
    while i < len(args) and args[i].startswith("-"):
        key, separator, _ = args[i].partition("=")
        if key not in values:
            return []
        i += 1 if separator else 2
    return args[i:]


def _gh_api_is_read(args: list[str]) -> bool:
    method = "GET"
    i = 1
    endpoint = ""
    while i < len(args):
        token = args[i]
        key, separator, value = token.partition("=")
        if key in {"-X", "--method"}:
            if not separator:
                i += 1
                if i >= len(args):
                    return False
                value = args[i]
            method = value.upper()
        elif token.startswith("-X") and token != "-X":
            method = token[2:].upper()
        elif key in {"-f", "-F", "--field", "--raw-field", "--input"} or token.startswith(
            ("-f", "-F")
        ):
            # Fields cause implicit POST; opaque request bodies/GraphQL mutations
            # require execution permission even with a forged GET flag.
            return False
        elif not token.startswith("-") and not endpoint:
            endpoint = token
        i += 1
    return method in {"GET", "HEAD"} and endpoint != "graphql"


def inspection_command(tokens: list[str]) -> bool | None:
    """Classify only explicitly owned complete inspection argument shapes."""
    if not tokens:
        return False
    tool, args = tokens[0], tokens[1:]
    if tool == "git":
        return _validate_git_readonly(tokens)[0]
    if tool == "sed":
        return args in [["--help"], ["--version"]] or _validate_sed_readonly(tokens)[0]
    if tool == "sort":
        return not any(
            (
                len(value) > 2
                and value.startswith("--")
                and any(
                    option.startswith(value.partition("=")[0])
                    for option in {"--output", "--compress-program"}
                )
            )
            or (value.startswith("-") and not value.startswith("--") and "o" in value[1:])
            for value in args
        )
    if tool == "rg":
        return not any(value.partition("=")[0] in {"--pre", "--hostname-bin"} for value in args)
    if tool == "find":
        index = 0
        while index < len(args):
            value = args[index]
            if value in FIND_VALUE_TESTS:
                index += 2
                continue
            if value in {
                "-delete",
                "-fprint",
                "-fprintf",
                "-fprint0",
                "-fls",
                "-exec",
                "-execdir",
                "-ok",
                "-okdir",
            }:
                return False
            index += 1
        return True
    if tool in _MEDIA_TOOLS:
        return _media_inspection(tool, args)
    if tool == "read":
        return _read_builtin_inspection(args)
    if tool == "printf":
        return _printf_builtin_inspection(args)
    if tool in {"date", "hostname", "uniq", "file", "xxd", "less", "more", "mktemp"}:
        return _utility_inspection(tool, args)
    if tool in {"tar", "unzip"}:
        return _archive_listing(tool, args)
    if tool in OPAQUE_PROGRAMS:
        return (
            tool not in {".", "source", "pytest", "tox"}
            and len(args) == 1
            and args[0] in {"--help", "--version", "-h", "-V", *(["-v"] if tool == "node" else [])}
        )
    if tool in {"ruff", "black"}:
        if len(args) == 1 and args[0] in {"--version", "--help", "-h"}:
            return True
        if any(arg.startswith(("--fix", "--unsafe-fix")) for arg in args):
            return False
        if not _valid_cli_options(tool, args):
            return False
        return (
            ("--check" in args or "--diff" in args)
            if tool == "black"
            else bool(args)
            and (
                args[0] == "check"
                or (args[0] == "format" and ("--check" in args or "--diff" in args))
            )
        )
    if tool not in _INSPECTION_PATHS:
        return None
    if len(args) == 1 and args[0] in {"--version", "--help", "-h", "-V"}:
        return True
    action_args = _action_args(tool, args)
    if tool == "gh" and action_args[:1] == ["api"]:
        return _valid_cli_options(tool, action_args[1:]) and _gh_api_is_read(action_args)
    if tool == "npm" and action_args == ["version"]:
        return True
    if tool == "npm" and action_args[:1] == ["audit"]:
        return _valid_cli_options(tool, action_args[1:], positionals=False)
    for path in sorted(_INSPECTION_PATHS[tool], key=len, reverse=True):
        if tuple(action_args[: len(path)]) != path:
            continue
        if (
            tool == "cargo"
            and path[0] in {"tree", "metadata"}
            and not any(option in action_args for option in ("--locked", "--frozen"))
        ):
            return False
        return _valid_cli_options(
            tool, action_args[len(path) :], positionals=path != ("cluster-info",)
        )
    return False


def execution_environment_assignment(token: str) -> bool:
    """Identify explicit assignments that can replace executable/config owners."""
    name, separator, _ = token.partition("=")
    return bool(separator) and (
        name
        in {
            "PATH",
            "IFS",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "BASH_ENV",
            "ENV",
            "PYTHONPATH",
            "PYTHONHOME",
            "NODE_OPTIONS",
            "RIPGREP_CONFIG_PATH",
            "GIT_EXTERNAL_DIFF",
            "GIT_SSH",
            "GIT_SSH_COMMAND",
            "GIT_EXEC_PATH",
            "RUSTC",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTDOC",
            "CARGO_BUILD_RUSTC",
            "CARGO_BUILD_RUSTC_WRAPPER",
            "GIT_PAGER",
            "PAGER",
            "LESSOPEN",
            "TAR_OPTIONS",
        }
        or name.startswith(("DYLD_", "GIT_CONFIG_"))
    )


def _archive_listing(tool: str, args: list[str]) -> bool:
    listing = False
    i = 0
    while i < len(args):
        argument = args[i]
        if argument == "--":
            break
        if tool == "unzip":
            if argument.startswith("-"):
                if len(argument) < 2 or argument[1:].strip("lZ1sv"):
                    return False
                listing |= "l" in argument or "Z" in argument
        elif argument in {"--list", "--verbose", "--gzip", "--bzip2", "--xz", "--zstd"}:
            listing |= argument == "--list"
        elif argument.startswith("--file="):
            pass
        elif argument in {"--file", "-f"}:
            i += 1
            if i >= len(args):
                return False
        elif argument.startswith("-") or (i == 0 and argument.isalpha()):
            flags = argument[1:] if argument.startswith("-") else argument
            if not flags:
                return False
            for index, flag in enumerate(flags):
                if flag not in "tvfzjJ":
                    return False
                listing |= flag == "t"
                if flag == "f":
                    if index + 1 == len(flags):
                        i += 1
                        if i >= len(args):
                            return False
                    break
        i += 1
    return listing


FIND_VALUE_TESTS = frozenset(
    {
        "-name",
        "-iname",
        "-lname",
        "-ilname",
        "-path",
        "-ipath",
        "-wholename",
        "-iwholename",
        "-regex",
        "-iregex",
        "-type",
        "-xtype",
        "-user",
        "-group",
        "-uid",
        "-gid",
        "-perm",
        "-newer",
        "-anewer",
        "-cnewer",
        "-size",
        "-inum",
        "-links",
        "-mtime",
        "-atime",
        "-ctime",
        "-maxdepth",
        "-mindepth",
        "-fstype",
        "-context",
    }
)


def dynamic_arguments_safe(tool: str, words: list[ShellWord]) -> bool:
    """Unknown argv may not choose effectful actions/options or separator arity."""
    sensitive = (
        set(_INSPECTION_PATHS)
        | OPAQUE_PROGRAMS
        | _MEDIA_TOOLS
        | {
            "git",
            "sort",
            "sed",
            "find",
            "env",
            "ssh",
            "ruff",
            "black",
            "tar",
            "unzip",
            "read",
            "printf",
            "rg",
            "date",
            "hostname",
            "uniq",
            "file",
            "xxd",
            "less",
            "more",
        }
    )
    if tool not in sensitive:
        return True
    if tool == "printf":
        index = _printf_format_index([word.value for word in words[1:]])
        return index is not None and index + 1 < len(words) and not words[index + 1].dynamic
    positional = False
    consumes_value = False
    separator_proven = True
    bool_flags = _CLI_BOOL_FLAGS.get(tool, frozenset()) | _DYNAMIC_BOOL_FLAGS.get(tool, frozenset())
    value_flags = _CLI_VALUE_FLAGS.get(tool, frozenset()) | _DYNAMIC_VALUE_FLAGS.get(
        tool, frozenset()
    )
    for word in words[1:]:
        if consumes_value:
            consumes_value = False
            if word.dynamic and (word.field_split or tool == "sed"):
                return False
            continue
        if tool == "find" and not word.dynamic and word.value in FIND_VALUE_TESTS:
            consumes_value = True
            continue
        if not word.dynamic and word.value in value_flags:
            consumes_value = True
            continue
        if not word.dynamic and word.value == "--":
            positional = separator_proven and tool != "find"
            continue
        if not word.dynamic:
            if (
                word.value.startswith("-")
                and word.value.partition("=")[0] not in bool_flags
                and word.value.partition("=")[0] not in value_flags
            ):
                # Even a valid unmodeled option may consume the following --.
                separator_proven = False
            continue
        if tool in {"env", "ssh"}:
            return False
        if positional:
            continue
        if not word.field_split and word.fixed_prefix.startswith(("./", "../", "/")):
            continue
        return False
    return True


def _media_inspection(tool: str, args: list[str]) -> bool:
    """Accept a small stdout inspection grammar, never arbitrary media programs."""
    if len(args) == 1 and args[0] in {"--help", "--version", "-h", "-help", "-v"}:
        return True
    if tool == "tesseract":
        if len(args) < 2 or args[1] != "stdout":
            return False
        index = 2
        while index < len(args):
            if args[index] not in {"-l", "--psm", "--oem", "--dpi"} or index + 1 >= len(args):
                return False
            index += 2
        return True
    flags: dict[str, set[str]] = {
        "ffmpeg": {"-hide_banner", "-nostdin", "-an", "-sn", "-dn"},
        "ffprobe": {
            "-hide_banner",
            "-show_format",
            "-show_streams",
            "-show_packets",
            "-show_frames",
            "-show_programs",
            "-show_chapters",
            "-show_error",
            "-count_frames",
            "-count_packets",
        },
        "pdftotext": {"-layout", "-raw", "-nopgbrk", "-table", "-simple"},
        "pdftocairo": {
            "-png",
            "-jpeg",
            "-tiff",
            "-singlefile",
            "-gray",
            "-mono",
            "-transp",
            "-cropbox",
        },
    }
    value_options: dict[str, set[str]] = {
        "ffmpeg": {
            "-v",
            "-loglevel",
            "-ss",
            "-i",
            "-frames:v",
            "-f",
            "-vcodec",
            "-c:v",
            "-pix_fmt",
        },
        "ffprobe": {
            "-v",
            "-loglevel",
            "-select_streams",
            "-show_entries",
            "-of",
            "-print_format",
            "-read_intervals",
            "-i",
        },
        "pdftotext": {
            "-f",
            "-l",
            "-enc",
            "-eol",
            "-x",
            "-y",
            "-W",
            "-H",
            "-r",
            "-fixed",
            "-colspacing",
        },
        "pdftocairo": {
            "-f",
            "-l",
            "-r",
            "-rx",
            "-ry",
            "-scale-to",
            "-scale-to-x",
            "-scale-to-y",
            "-x",
            "-y",
            "-W",
            "-H",
        },
    }
    positionals: list[str] = []
    values: list[tuple[str, str]] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in flags[tool]:
            pass
        elif argument in value_options[tool]:
            index += 1
            if index >= len(args):
                return False
            values.append((argument, args[index]))
        elif argument.startswith("-") and argument != "-":
            return False
        else:
            positionals.append(argument)
        index += 1
    if tool == "ffmpeg":
        return (
            positionals == ["-"]
            and sum(option == "-i" for option, _ in values) == 1
            and any(option == "-f" and value == "image2pipe" for option, value in values)
            and all(value == "image2pipe" for option, value in values if option == "-f")
        )
    if tool == "ffprobe":
        return len(positionals) + sum(option == "-i" for option, _ in values) == 1
    return (
        len(positionals) == 2
        and positionals[-1] == "-"
        and (tool == "pdftotext" or "-singlefile" in args)
    )


_MEDIA_TOOLS = frozenset({"ffmpeg", "ffprobe", "tesseract", "pdftotext", "pdftocairo"})
_CLI_BOOL_FLAGS: dict[str, frozenset[str]] = {
    "gh": frozenset(
        {
            "--comments",
            "--watch",
            "--exit-status",
            "--verbose",
            "--include",
            "--silent",
            "--paginate",
            "--slurp",
            "-c",
            "-i",
        }
    ),
    "docker": frozenset(
        {
            "--all",
            "--quiet",
            "--no-trunc",
            "--no-stream",
            "--no-cache",
            "--services",
            "--volumes",
            "--images",
            "--names",
            "--no-interpolate",
            "--hash",
            "-a",
            "-q",
        }
    ),
    "helm": frozenset(
        {
            "--all",
            "--all-namespaces",
            "--short",
            "--deployed",
            "--failed",
            "--pending",
            "--uninstalled",
            "--superseded",
            "-a",
            "-A",
            "-q",
        }
    ),
    "kubectl": frozenset(
        {
            "--all-namespaces",
            "--show-labels",
            "--no-headers",
            "--watch",
            "--watch-only",
            "--follow",
            "--previous",
            "--timestamps",
            "--all-containers",
            "--insecure-skip-tls-verify",
            "-A",
            "-w",
            "-f",
            "-p",
        }
    ),
    "az": frozenset({"--all", "--only-show-errors", "--verbose"}),
    "pip": frozenset(
        {
            "--local",
            "--user",
            "--editable",
            "--not-required",
            "--exclude-editable",
            "--include-editable",
            "--outdated",
            "--uptodate",
            "--disable-pip-version-check",
            "--isolated",
            "--no-input",
        }
    ),
    "npm": frozenset({"--json", "--long", "--all", "--parseable", "--global", "-g", "-l"}),
    "yarn": frozenset({"--json", "--offline"}),
    "pnpm": frozenset({"--json", "--long", "--global", "-g"}),
    "cargo": frozenset(
        {
            "--locked",
            "--frozen",
            "--offline",
            "--no-deps",
            "--all-features",
            "--no-default-features",
        }
    ),
    "go": frozenset({"-m", "-v"}),
    "ip": frozenset({"-j", "-json", "-o", "-oneline", "-4", "-6"}),
    "ruff": frozenset({"--check", "--diff", "--no-cache", "--quiet", "-q"}),
    "black": frozenset({"--check", "--diff", "--color", "--no-color", "--quiet", "-q"}),
}
_CLI_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "gh": frozenset(
        {
            "--json",
            "--jq",
            "--template",
            "--limit",
            "--state",
            "--author",
            "--assignee",
            "--label",
            "--search",
            "--repo",
            "--hostname",
            "--method",
            "--header",
            "--branch",
            "--workflow",
            "-L",
            "-R",
            "-q",
            "-t",
            "-X",
            "-H",
        }
    ),
    "docker": frozenset(
        {"--format", "--filter", "--tail", "--since", "--until", "--context", "--host", "-f", "-H"}
    ),
    "helm": frozenset(
        {"--output", "--namespace", "--filter", "--max", "--offset", "--kube-context", "-o", "-n"}
    ),
    "kubectl": frozenset(
        {
            "--output",
            "--namespace",
            "--context",
            "--cluster",
            "--user",
            "--server",
            "--selector",
            "--field-selector",
            "--sort-by",
            "--tail",
            "--since",
            "--since-time",
            "--container",
            "--request-timeout",
            "-o",
            "-n",
            "-l",
            "-c",
            "-s",
        }
    ),
    "az": frozenset(
        {
            "--output",
            "--query",
            "--subscription",
            "--resource-group",
            "--name",
            "--ids",
            "-o",
            "-g",
            "-n",
        }
    ),
    "pip": frozenset({"--format", "--path", "--exclude"}),
    "npm": frozenset({"--depth", "--workspace", "-w"}),
    "yarn": frozenset({"--depth", "--pattern"}),
    "pnpm": frozenset({"--depth", "--filter"}),
    "cargo": frozenset(
        {"--format-version", "--manifest-path", "--features", "--target", "--prefix", "--depth"}
    ),
    "ruff": frozenset(
        {"--output-format", "--select", "--ignore", "--target-version", "--line-length"}
    ),
    "black": frozenset({"--line-length", "--target-version", "--include", "--exclude", "-l", "-t"}),
}
_CLI_BOOL_FLAGS["podman"] = _CLI_BOOL_FLAGS["docker"]
_CLI_VALUE_FLAGS["podman"] = _CLI_VALUE_FLAGS["docker"]
_CLI_BOOL_FLAGS["pip3"] = _CLI_BOOL_FLAGS["pip"]
_CLI_VALUE_FLAGS["pip3"] = _CLI_VALUE_FLAGS["pip"]
_DYNAMIC_BOOL_FLAGS = {
    "git": frozenset(
        {
            "--no-pager",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--stat",
            "--name-only",
            "--name-status",
            "--quiet",
            "--exit-code",
            "--oneline",
            "--porcelain",
        }
    ),
    "sort": frozenset({"-n", "-r", "-u", "-f", "-b", "-h", "-V"}),
    "sed": frozenset({"-n", "-E", "-r"}),
    "rg": frozenset({"-n", "-i", "-F", "-l"}),
}
_DYNAMIC_VALUE_FLAGS = {
    "git": frozenset(
        {"--src-prefix", "--dst-prefix", "--format", "--pretty", "--git-dir", "--work-tree", "-C"}
    ),
    "sed": frozenset({"-e", "--expression", "-f"}),
    "tar": frozenset({"-f", "--file"}),
}


def _valid_cli_options(tool: str, args: list[str], *, positionals: bool = True) -> bool:
    """Validate all trailing options using one finite option/arity capability table."""
    booleans = _CLI_BOOL_FLAGS.get(tool, frozenset())
    values = _CLI_VALUE_FLAGS.get(tool, frozenset())
    index = 0
    while index < len(args):
        argument = args[index]
        key, separator, value = argument.partition("=")
        if argument == "--":
            return positionals or index + 1 == len(args)
        if key in booleans:
            if separator and value not in {"true", "false"}:
                return False
        elif key in values:
            if not separator:
                index += 1
                if index >= len(args):
                    return False
        elif argument.startswith("-"):
            if argument.startswith("--"):
                return False
            offset = 1
            while offset < len(argument):
                flag = "-" + argument[offset]
                if flag in booleans:
                    offset += 1
                elif flag in values:
                    if offset + 1 == len(argument):
                        index += 1
                        if index >= len(args):
                            return False
                    break
                else:
                    return False
        elif not positionals:
            return False
        index += 1
    return True


def _safe_receiver(name: str) -> bool:
    return re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", name
    ) is not None and not execution_environment_assignment(name + "=")


def _read_builtin_inspection(args: list[str]) -> bool:
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            return all(_safe_receiver(value) for value in args[index + 1 :])
        if argument.startswith("-"):
            offset = 1
            while offset < len(argument):
                flag = argument[offset]
                if flag in "rseE":
                    offset += 1
                    continue
                if flag not in "adnNptui":
                    return False
                value = argument[offset + 1 :]
                if not value:
                    index += 1
                    if index >= len(args):
                        return False
                    value = args[index]
                if flag == "a" and not _safe_receiver(value):
                    return False
                break
        elif not _safe_receiver(argument):
            return False
        index += 1
    return True


def _printf_format_index(args: list[str]) -> int | None:
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            return index + 1 if index + 1 < len(args) else None
        if not argument.startswith("-v"):
            return index
        target = argument[2:]
        if not target:
            index += 1
            if index >= len(args):
                return None
            target = args[index]
        if not _safe_receiver(target):
            return None
        index += 1
    return None


def _printf_builtin_inspection(args: list[str]) -> bool:
    if args == ["--help"]:
        return True
    index = _printf_format_index(args)
    if index is None:
        return False
    format_text = args[index]
    offset = 0
    while offset < len(format_text):
        if format_text[offset] != "%":
            offset += 1
        elif format_text.startswith("%%", offset):
            offset += 2
        else:
            conversion = re.match(
                r"%[-+ #0]*(?:[0-9]+|\*)?(?:\.(?:[0-9]+|\*)?)?[hlLzjt]?([diouxXfFeEgGaAcCsSbqQT])",
                format_text[offset:],
            )
            if conversion is None:
                return False
            offset += conversion.end()
    return True


PURE_OUTPUT_COMMANDS = frozenset(
    {
        "pwd",
        "ls",
        "ll",
        "la",
        "cd",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "egrep",
        "fgrep",
        "cut",
        "tr",
        "paste",
        "join",
        "tac",
        "nl",
        "basename",
        "dirname",
        "realpath",
        "readlink",
        "which",
        "whereis",
        "type",
        "find",
        "sort",
        "rg",
        "expr",
        "test",
        "[",
        "true",
        "false",
        ":",
        "echo",
        "shift",
        "break",
        "continue",
        "return",
        "stat",
        "md5sum",
        "sha256sum",
        "shasum",
        "strings",
        "hexdump",
        "od",
        "diff",
        "cmp",
        "zcat",
        "gzcat",
        "uname",
        "whoami",
        "id",
        "uptime",
        "df",
        "du",
        "free",
        "ps",
        "pgrep",
        "lsof",
        "printenv",
        "zipinfo",
        "pdfinfo",
    }
)


def _utility_inspection(tool: str, args: list[str]) -> bool:
    if tool == "mktemp":
        return args in [["--help"], ["--version"]]
    if tool == "hostname":
        return not args or all(
            value in {"-s", "-f", "-d", "-i", "-I", "-a", "-A"} for value in args
        )
    if tool == "date":
        return all(
            value in {"-u", "--utc", "--iso-8601", "-I", "-R", "--rfc-email"}
            or value.startswith("+")
            for value in args
        )
    if tool in {"less", "more"}:
        return all(
            not value.startswith("+")
            and (
                not value.startswith("-")
                or value in {"-F", "-R", "-S", "-N", "-n", "-X", "-e", "-E", "-Q", "-q"}
            )
            for value in args
        )
    flags = {
        "uniq": {"-c", "-d", "-D", "-u", "-i", "-z"},
        "file": {
            "-b",
            "-i",
            "-L",
            "-z",
            "-0",
            "-N",
            "-r",
            "-s",
            "--brief",
            "--mime",
            "--mime-type",
            "--mime-encoding",
        },
        "xxd": {"-p", "-a", "-b", "-u"},
    }[tool]
    value_flags = {"uniq": {"-f", "-s", "-w"}, "file": set(), "xxd": {"-c", "-g", "-l", "-s"}}[tool]
    files = 0
    index = 0
    positional = False
    while index < len(args):
        value = args[index]
        if value == "--" and not positional:
            positional = True
        elif value in flags and not positional:
            pass
        elif value in value_flags and not positional:
            index += 1
            if index >= len(args):
                return False
        elif value.startswith("-") and value != "-" and not positional:
            return False
        else:
            files += 1
        index += 1
    return tool == "file" or files <= 1


def cacheable_inspection(command: str) -> bool:
    """Cache only literal process-stable metadata, never file/network snapshots."""
    from victor.security.shell_syntax import (
        scan_substitutions,
        shell_words,
        split_shell_commands,
        shell_assignments,
    )

    if any(character in command for character in "<>|;&\n"):
        return False
    try:
        children, sanitized = scan_substitutions(command)
        if children or len(split_shell_commands(sanitized)) != 1 or shell_assignments(sanitized):
            return False
        words = shell_words(sanitized)
    except ValueError:
        return False
    if not words or any(word.dynamic for word in words):
        return False
    tool = words[0].value
    args = [word.value for word in words[1:]]
    if tool == "echo":
        return True
    if tool == "whoami":
        return not args
    if tool == "id":
        return not args or all(value in {"-u", "-g", "-G"} for value in args)
    if tool == "uname":
        return all(
            value in {"-a", "-s", "-n", "-r", "-v", "-m", "-p", "-i", "-o"} for value in args
        )
    return False


GIT_READONLY_SUBCOMMANDS: set[str] = {
    "status",
    "log",
    "show",
    "diff",
    "branch",
    "tag",
    "remote",
    "ls-files",
    "ls-tree",
    "rev-parse",
    "rev-list",
    "describe",
    "shortlog",
    "blame",
    "grep",
    "config",  # readonly by default (no --global, --unset)
    "reflog",
    "stash",  # list only
    "worktree",  # list only
    "cat-file",
    "ls-remote",
    "check-ignore",
    "check-attr",
    "name-rev",
    "verify-commit",
    "verify-tag",
    "for-each-ref",
}


def _sed_delimited_end(program: str, start: int) -> int:
    delimiter = program[start]
    i = start + 1
    while i < len(program):
        if program[i] == "\\":
            i += 2
        elif program[i] == delimiter:
            return i + 1
        else:
            i += 1
    return -1


def _readonly_sed_program(program: str) -> bool:
    """Recognize bounded display/substitution syntax; other sed programs need writes."""
    if not program or "\n" in program:
        return False
    # Simple address selection covers the file-viewing path without evaluating
    # arbitrary sed programs (e/r/w and substitution e/w flags can execute/write).
    position = 0
    address = re.match(r"(?:[0-9]+|\$)(?:,(?:[0-9]+|\$))?", program)
    if address:
        position = address.end()
    elif program.startswith("/"):
        position = _sed_delimited_end(program, 0)
        if position < 0:
            return False
    rest = program[position:].strip()
    if rest in {"p", "d", "q"}:
        return True
    if len(rest) < 4 or rest[0] != "s" or rest[1].isalnum() or rest[1] == "\\":
        return False
    first_end = _sed_delimited_end(rest, 1)
    if first_end < 0:
        return False
    # The replacement starts after the pattern delimiter; parse its escaped
    # characters using the same delimiter rather than splitting on shell text.
    replacement_end = _sed_delimited_end(rest, first_end - 1)
    return (
        replacement_end >= 0 and re.fullmatch(r"[gpIiMm0-9]*", rest[replacement_end:]) is not None
    )


def _validate_sed_readonly(tokens: list[str]) -> tuple[bool, str]:
    programs: list[str] = []
    i = 1
    options = True
    while i < len(tokens):
        token = tokens[i]
        if options and token == "--":
            options = False
        elif options and token in {"-e", "--expression"}:
            i += 1
            if i >= len(tokens):
                return False, "sed (missing program)"
            programs.append(tokens[i])
        elif options and token.startswith("--expression="):
            programs.append(token.partition("=")[2])
        elif options and token.startswith("-e"):
            programs.append(token[2:])
        elif options and token.startswith("-"):
            if token not in {"--quiet", "--silent", "--regexp-extended"} and not (
                len(token) > 1 and token[1:].strip("nEr") == ""
            ):
                return False, "sed (unsupported option)"
        elif not programs:
            programs.append(token)
        i += 1
    if not programs or not all(_readonly_sed_program(program) for program in programs):
        return False, "sed (unsupported program)"
    return True, ""


def _readonly_ref_listing(args: list[str], short_flags: str) -> bool:
    listing = "--list" in args or "-l" in args
    for arg in args:
        if arg in {"--list", "--all", "--remotes", "--verbose", "--show-current", "--no-color"}:
            continue
        if arg.startswith(("--format=", "--sort=", "--color=")):
            continue
        if arg.startswith("-"):
            if len(arg) <= 1 or arg[1:].strip(short_flags) != "":
                return False
        elif not listing:
            return False
    return True


def _validate_git_readonly(tokens: list[str]) -> tuple[bool, str]:
    """Validate global options and the actual effect of nested git operations."""
    i = 1
    while i < len(tokens) and tokens[i].startswith("-"):
        token = tokens[i]
        if token in {"-C", "--git-dir", "--work-tree"}:
            i += 2
        elif token.startswith(("-C", "--git-dir=", "--work-tree=")) or token in {
            "--no-pager",
            "--no-optional-locks",
            "--literal-pathspecs",
            "--no-replace-objects",
        }:
            i += 1
        else:
            return False, "git (unsupported global option)"
    if i >= len(tokens):
        return False, "git (missing subcommand)"
    subcommand, args = tokens[i], tokens[i + 1 :]
    if subcommand not in GIT_READONLY_SUBCOMMANDS:
        return False, f"git {subcommand}"
    if any(
        (
            len(arg) > 2
            and arg.startswith("--")
            and any(
                dangerous.startswith(arg.partition("=")[0])
                for dangerous in {"--output", "--ext-diff", "--textconv", "--open-files-in-pager"}
            )
        )
        or (subcommand == "grep" and arg.startswith("-O"))
        for arg in args
    ):
        return False, f"git {subcommand} (output or executable helper)"
    safe = True
    if subcommand in {"branch", "tag"}:
        safe = _readonly_ref_listing(args, "alrv" if subcommand == "branch" else "ln")
    elif subcommand == "remote":
        safe = not args or args == ["-v"] or args == ["--verbose"] or args[0] in {"show", "get-url"}
    elif subcommand == "stash":
        safe = bool(args) and args[0] in {"list", "show"}
    elif subcommand == "worktree":
        safe = bool(args) and args[0] == "list"
    elif subcommand == "reflog":
        safe = not args or args[0] in {"show", "exists"}
    elif subcommand == "config":
        forbidden = {
            "--add",
            "--replace-all",
            "--unset",
            "--unset-all",
            "--rename-section",
            "--remove-section",
            "--edit",
            "-e",
        }
        reading = {"--get", "--get-all", "--get-regexp", "--get-urlmatch", "--list", "-l"}
        safe = bool(args) and (args[0] in {"get", "list"} or bool(reading.intersection(args)))
        safe = safe and not any(arg.partition("=")[0] in forbidden for arg in args)
    return (True, "") if safe else (False, f"git {subcommand} (write operation)")


def inplace_sed_targets(tokens: list[str]) -> list[str] | None:
    """Contain literal simple in-place substitutions; opaque sed remains execution."""
    if len(tokens) < 4 or tokens[1] != "-i" or not _readonly_sed_program(tokens[2]):
        return None
    targets = tokens[3:]
    if any(not target or target.startswith("-") for target in targets):
        return None
    return targets
