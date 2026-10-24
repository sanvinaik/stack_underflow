# scripts/extract_eval.py
import json, sys
nb = json.load(open("../data/Train/Evaluation/Evaluation.ipynb", encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]

# First run with no args: list code cells so you can pick the right ones
if len(sys.argv) == 1:
    for i, c in enumerate(code):
        src = "".join(c["source"])
        print(f"--- code cell {i} ({len(src)} chars) ---\n{src[:200]}\n")
    sys.exit()

picked = [int(i) for i in sys.argv[1:]]          # e.g. python scripts/extract_eval.py 1 2 3
out = "\n\n".join("".join(code[i]["source"]) for i in picked)
open("../eval/official_eval.py", "w", encoding="utf-8", newline="").write(out)