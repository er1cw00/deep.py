from collections import namedtuple
from typing import List, Literal

FaceAnalyserOrder = Literal['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']

Face = namedtuple('Face',
[
	'bbox',
	'landmarks',
	'score',
])