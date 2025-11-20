from collections import deque
from pathlib import Path
path = Path('backtest.log')
last = deque(maxlen=200)
with path.open('r', encoding='utf-8') as f:
    for line in f:
        last.append(line.rstrip('\n'))
print('\n'.join(last))
