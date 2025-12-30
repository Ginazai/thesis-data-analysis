from pathlib import Path
from analysis import parse_log_lines, compute_inference_duration

p = Path("logs/escenario 3/MOBILE - hib.txt")
text = p.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()
events = parse_log_lines(lines)

print("First 30 events with timestamps and event names:")
for i, ev in enumerate(events[:40]):
    print(i, ev.get('ts'), ev.get('event'))

compute = compute_inference_duration(events)
print('\nComputed duration (ms):', compute)

# show durations assigned to events if any
assigned = [(i, ev.get('event'), ev.get('ts'), ev.get('duration_ms')) for i, ev in enumerate(events[:60]) if ev.get('duration_ms') is not None]
print('\nAssigned durations (index, event, ts, duration_ms):')
for it in assigned:
    print(it)
