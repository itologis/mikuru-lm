#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
binary = Path(sys.argv[1]).resolve()
experiment = Path(__file__).resolve().parents[1]

def ask(prompt):
    p = subprocess.run(
        [str(binary)], input=prompt + "\n/quit\n",
        text=True, capture_output=True, cwd=experiment, check=True
    )
    match = None
    replies = []
    for line in p.stdout.splitlines():
        if line.startswith("Matched prompt: "):
            match = line[len("Matched prompt: "):]
        elif line == "Match: none":
            match = None
        if line.startswith("Mikuru> "):
            replies.append(line[len("Mikuru> "):])
    return match, replies[0] if replies else "", p.stdout

def check(ok, name, output):
    if not ok:
        print("FAIL:", name)
        print(output)
        raise SystemExit(1)
    print("PASS:", name)

m,r,o = ask("Could you explain neural networks in beginner-friendly language?")
check(m and "neural network" in m.lower(), "neural-network paraphrase", o)
m,r,o = ask("Can you explain a neural network simply?")
check(m != "Can you explain more simply?", "topic beats generic wording", o)
m,r,o = ask("How can I improve this dialogue model?")
check(m != "What should we improve first?", "reject generic improvement match", o)
m,r,o = ask("I feel exhausted after a difficult day at school.")
check(m != "I finished something difficult." and "LET'S GO" not in r.upper(), "exhaustion avoids celebration", o)
m,r,o = ask("What is the difference between a compiler and an interpreter?")
check(m != "I got a compiler error", "comparison avoids troubleshooting", o)
m,r,o = ask("I failed my exam, and I'm disappointed in myself.")
check(m != "I passed my exam", "failure avoids passing response", o)
m,r,o = ask("Help me write a program about a dragon.")
check(m != "I want to write a story.", "coding avoids story advice", o)
m,r,o = ask("Help me write a story about a dragon.")
check(m is not None and "story" in m.lower(), "story advice remains available", o)
m,r,o = ask("What should we improve first?")
check(m == "What should we improve first?", "exact prompt preserved", o)
print("All Retrieval V1.1 regression checks passed.")
