from __future__ import annotations

import argparse
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".log", ".csv", ".json"}


def decode_best(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data:
        try:
            return data.decode("utf-16-le", errors="replace")
        except Exception:
            pass
    for enc in ("utf-8", "gb18030", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def normalize_file(path: Path) -> tuple[int, int, int]:
    raw = path.read_bytes()
    txt = clean_text(decode_best(raw))
    path.write_text(txt, encoding="utf-8", newline="\n")
    out = path.read_bytes()
    return len(raw), len(out), out.count(b"\x00")


def should_handle(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_EXT


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize output files to UTF-8 + LF + no NUL")
    parser.add_argument("target", nargs="?", default="outputs", help="folder to normalize")
    parser.add_argument("--check", action="store_true", help="only check, do not rewrite")
    args = parser.parse_args()

    root = Path(args.target)
    if not root.exists():
        print(f"target_not_found: {root}")
        return

    files = [p for p in root.rglob("*") if should_handle(p)]
    files.sort()

    bad = []
    for p in files:
        data = p.read_bytes()
        nul = data.count(b"\x00")
        try:
            data.decode("utf-8")
            utf8_ok = True
        except Exception:
            utf8_ok = False
        if nul > 0 or not utf8_ok or b"\r\n" in data:
            bad.append((p, nul, utf8_ok, b"\r\n" in data))

    if args.check:
        print(f"checked={len(files)}")
        print(f"non_compliant={len(bad)}")
        for p, nul, utf8_ok, has_crlf in bad:
            print(f"{p}\tNUL={nul}\tutf8_ok={utf8_ok}\tcrlf={has_crlf}")
        return

    changed = 0
    for p in files:
        before, after, nul = normalize_file(p)
        changed += 1
        print(f"normalized\t{p}\t{before}->{after}\tNUL={nul}")

    print(f"done\tfiles={len(files)}\tnormalized={changed}")


if __name__ == "__main__":
    main()
