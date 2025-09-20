#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs' / 'baselines'

def main():
    count_total = 0
    count_modified = 0
    for path in BASE.rglob('*.yaml'):
        count_total += 1
        text = path.read_text(encoding='utf-8')
        if 'data/processed' in text:
            new_text = text.replace('data/processed', 'data/preprocessed')
            if new_text != text:
                path.write_text(new_text, encoding='utf-8')
                count_modified += 1
                print(f"Updated: {path.relative_to(ROOT)}")
    print(f"Done. {count_modified}/{count_total} files updated.")

if __name__ == '__main__':
    main()


