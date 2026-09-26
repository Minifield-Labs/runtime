#!/usr/bin/env python3
"""Check standalone repository paths and explicit core configuration (Python 3.11+)."""

import argparse
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import tomllib

ENV_READS = r"(?:var|var_os|vars|vars_os|set_var|remove_var)"
PERSONAL_PATH = re.compile(r"/" + r"Users/[^\s`\"'<>)]*")
INLINE_LINK = re.compile(
    r"\]\(\s*(?P<target><[^>\n]+>|(?:\\.|[^\s()]|\([^\s()]*\))+)(?:\s+[^\n)]*)?\s*\)"
)
REFERENCE_LINK = re.compile(
    r"(?m)^ {0,3}\[(?!\^)[^]\n]+\]:\s*(?P<target><[^>\n]+>|\S+)"
)
RAW_LITERAL = re.compile(r'(?:br|cr|r)(#{0,255})"')
CHAR_LITERAL = re.compile(r"'(?:\\u\{[0-9a-fA-F_]+\}|\\x[0-9a-fA-F]{2}|\\.|[^'\\\n])'")


def blank(text: str) -> str:
    return "".join("\n" if character == "\n" else " " for character in text)


def rust_code(source: str) -> str:
    """Mask comments/literals, retaining positions and nested block comments."""
    output = list(source)
    index = 0
    while index < len(source):
        start = index
        if source.startswith("//", index):
            end = source.find("\n", index)
            index = len(source) if end == -1 else end
        elif source.startswith("/*", index):
            index += 2
            depth = 1
            while index < len(source) and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
        elif raw := RAW_LITERAL.match(source, index):
            closing = '"' + raw[1]
            end = source.find(closing, index + len(raw[0]))
            index = len(source) if end == -1 else end + len(closing)
        elif source[index] == '"':
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
        elif char := CHAR_LITERAL.match(source, index):
            index += len(char[0])
        else:
            index += 1
            continue
        output[start:index] = blank(source[start:index])
    code = "".join(output)
    # Exclude complete test items, including inline modules with arbitrary names.
    for match in reversed(
        list(re.finditer(r"#\s*\[\s*(?:cfg\s*\(\s*test\s*\)|test)\s*\]", code))
    ):
        item = re.search(r"[;{]", code[match.end() :])
        if item is None:
            continue
        end = match.end() + item.end()
        if code[end - 1] == "{":
            depth = 1
            while end < len(code) and depth:
                depth += (code[end] == "{") - (code[end] == "}")
                end += 1
        code = code[: match.start()] + blank(code[match.start() : end]) + code[end:]
    return code


def markdown_prose(source: str) -> str:
    source = re.sub(
        r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[^\n]*$",
        lambda match: blank(match[0]),
        source,
    )
    return re.sub(r"(`+)(?:(?!\1)[\s\S])*?\1", lambda match: blank(match[0]), source)


def audit(root: Path, files: list[Path]) -> list[str]:
    errors = []

    def report(path: Path, text: str, position: int, message: str) -> None:
        line = text.count("\n", 0, position) + 1
        errors.append(f"{path.relative_to(root)}:{line}: {message}")

    def cargo_paths(
        path: Path, text: str, value: dict, trail: tuple[str, ...] = ()
    ) -> None:
        dependency = any(
            part in {"dependencies", "dev-dependencies", "build-dependencies"}
            for part in trail
        ) or (trail and trail[0] in {"patch", "replace"})
        if dependency and isinstance(value.get("path"), str):
            target = (path.parent / value["path"]).resolve()
            literal = re.search(
                r"\bpath\s*=\s*['\"]" + re.escape(value["path"]) + r"['\"]", text
            )
            position = literal.start() if literal else 0
            if not target.is_relative_to(root) or not (
                target / "Cargo.toml"
            ).resolve().is_relative_to(root):
                report(
                    path,
                    text,
                    position,
                    f"Cargo dependency {'.'.join(trail)} leaves repository: {value['path']}",
                )
            elif not (target / "Cargo.toml").is_file():
                report(
                    path,
                    text,
                    position,
                    f"Cargo dependency {'.'.join(trail)} has no Cargo.toml: {value['path']}",
                )
        for name, child in value.items():
            if isinstance(child, dict):
                cargo_paths(path, text, child, (*trail, name))

    for path in sorted(files):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root)
        if path.name == "Cargo.toml":
            try:
                cargo_paths(path, text, tomllib.loads(text))
            except tomllib.TOMLDecodeError as error:
                report(path, text, 0, f"invalid Cargo TOML: {error}")
        if (
            path.suffix == ".rs"
            and relative.parts[0] == "crates"
            and "src" in relative.parts
            and relative.parts[1] != "infer-cli"
            and "tests" not in relative.parts
            and path.name not in {"tests.rs", "test.rs"}
        ):
            code = rust_code(text)
            aliases = {"env"}
            aliases.update(re.findall(r"use\s+std\s*::\s*env\s+as\s+(\w+)", code))
            aliases.update(
                re.findall(r"use\s+std\s*::\s*\{[^;]*?\benv\s+as\s+(\w+)", code)
            )
            aliases.update(
                re.findall(r"\benv\s*::\s*\{[^;]*?\bself\s+as\s+(\w+)", code)
            )
            modules = "|".join(re.escape(alias) for alias in sorted(aliases))
            pattern = re.compile(
                r"\b(?:std\s*::\s*)?(?:"
                + modules
                + r")\s*::\s*(?:"
                + ENV_READS
                + r"\b|\{[^}]*\b"
                + ENV_READS
                + r"\b)"
            )
            for match in pattern.finditer(code):
                report(
                    path,
                    text,
                    match.start(),
                    "runtime environment configuration in core source",
                )
        if path.suffix.lower() == ".md":
            prose = markdown_prose(text)
            for expression in (INLINE_LINK, REFERENCE_LINK):
                for match in expression.finditer(prose):
                    target = html.unescape(
                        match["target"].removeprefix("<").removesuffix(">")
                    )
                    target = re.sub(r"\\([\\`*{}\[\]()#+.!_<> -])", r"\1", target)
                    parsed = urlsplit(target)
                    if (
                        parsed.scheme
                        or parsed.netloc
                        or not parsed.path
                        or parsed.path.startswith("/")
                    ):
                        continue
                    destination = (path.parent / unquote(parsed.path)).resolve()
                    if (
                        not destination.is_relative_to(root)
                        or not destination.is_file()
                    ):
                        report(
                            path,
                            text,
                            match.start(),
                            f"Markdown target isn't a repository file: {target}",
                        )
        if relative.parts[:2] != ("docs", "research"):
            for match in PERSONAL_PATH.finditer(text):
                report(
                    path,
                    text,
                    match.start(),
                    "personal absolute path in active repository content",
                )
    return errors


def repository_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / name.decode("utf-8")
        for name in sorted(set(result.stdout.split(b"\0")))
        if name and (root / name.decode("utf-8")).is_file()
    ]


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="minifield-repository-check-") as temporary:
        root = Path(temporary).resolve()
        sources = {
            "Cargo.toml": '[dependencies]\nlocal = { path = "crates/core" }\n',
            "crates/core/Cargo.toml": '[package]\nname = "core"\nversion = "0.1.0"\n',
            "crates/core/src/lib.rs": '#[cfg(test)]\nmod arbitrary { fn oracle() { std::env::var("ORACLE"); } }\nconst DOC: &str = r#"std::env::var("EXAMPLE")"#;\n/* nested /* std::env::var */ comment */\n',
            "README.md": "[core](crates/core/Cargo.toml)\n[external](https://example.com)\n[fragment](#example)\n",
        }
        for name, text in sources.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        files = [root / name for name in sources]
        assert audit(root, files) == []
        (root / "Cargo.toml").write_text(
            '[dependencies]\nmissing = { path = "absent" }\noutside = { path = ".." }\n'
        )
        (root / "crates/core/src/lib.rs").write_text(
            'fn live() { std::env::var("LIVE"); }\n'
        )
        (root / "README.md").write_text(
            "[missing](absent.md)\n[ref]: absent-ref.md\n"
            + "/"
            + "Users/developer/private/model\n"
        )
        errors = audit(root, files)
        assert len(errors) == 6, errors
        assert sum("Cargo dependency" in error for error in errors) == 2
        assert sum("runtime environment" in error for error in errors) == 1
        assert sum("Markdown target" in error for error in errors) == 2
        assert sum("personal absolute" in error for error in errors) == 1
    print("repository checker self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return 0
    root = arguments.repo_root.resolve(strict=True)
    try:
        files = repository_files(root)
        errors = audit(root, files)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"repository check couldn't run: {error}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"repository check failed: {len(errors)} violation(s)", file=sys.stderr)
        return 1
    print(f"repository check passed ({len(files)} non-ignored files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
