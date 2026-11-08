"""Speech-core boundary used by the lab's edges/2 rule, from a 10 ms level track.

Derived from cantonese-speech-analysis scripts/analysis/canto/steps.py, Apache-2.0.
Only core selection is implemented here, not the edge-event detector.
"""

def segments(levels, merge_s):
    if not levels:
        return []
    threshold = max(levels) - 40.0
    spans = []
    start = None
    for i, level in enumerate(levels):
        if level > threshold and start is None:
            start = i
        elif level <= threshold and start is not None:
            spans.append([start, i])
            start = None
    if start is not None:
        spans.append([start, len(levels)])
    merged = []
    for start, end in spans:
        if merged and (start - merged[-1][1]) * 0.01 < merge_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(round(a * 0.01, 3), round(b * 0.01, 3)) for a, b in merged if (b - a) * 0.01 >= 0.04]


def speech_core(levels):
    if not segments(levels, 0.15):
        return None
    spans = segments(levels, 0.05)
    if not spans:
        return None
    peaks = [max(levels[int(a / 0.01):max(int(a / 0.01) + 1, int(b / 0.01))]) for a, b in spans]
    candidates = [i for i, (a, b) in enumerate(spans) if b - a >= 0.1]
    if not candidates:
        candidates = [max(range(len(peaks)), key=peaks.__getitem__)]
    reference = max(peaks[i] for i in candidates)
    candidates = [i for i in candidates if peaks[i] >= reference - 18.0]
    groups = [[candidates[0]]]
    for i in candidates[1:]:
        if spans[i][0] - spans[groups[-1][-1]][1] < 0.4:
            groups[-1].append(i)
        else:
            groups.append([i])
    loudest = max(candidates, key=peaks.__getitem__)
    keep = [group for group in groups if loudest in group or spans[group[-1]][1] - spans[group[0]][0] >= 0.5]
    return spans[keep[0][0]][0], spans[keep[-1][-1]][1]
