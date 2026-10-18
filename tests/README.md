# Smoke tests

Run make test from the repository root. The target builds the executables, trains a fresh small model, starts the chat program with scripted commands, and checks that control commands work and reserved tokens are not printed. Logs are written to the ignored build/ directory.
