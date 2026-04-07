# Pipeline Directory

## Log Output

- **Never swallow log output.** Do not pipe long-running commands through `tail`, `head`, `grep`, or any filter that
  buffers or discards streaming output. Progress logs exist to be read in real time. If you need the last N lines, read
  the log file after the process finishes.
- When running pipeline scripts in the background, use `run_in_background` or redirect to a file — never `| tail -N`.
