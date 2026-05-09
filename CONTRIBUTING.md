# Contributing

This repository is currently a thesis-code cleanup draft. Before accepting external contributions, define:

1. canonical pipeline entry points;
2. coding style and test commands;
3. expected input/output schemas;
4. license and citation requirements.

For internal development, run:

```bash
bash tools/smoke_test.sh
python tools/audit_hardcoded_paths.py
```
