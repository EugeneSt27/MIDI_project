from collections import defaultdict

def onset_density(notes, tpb):
    beats = defaultdict(int)
    for st, _, _, _ in notes:
        beat = int(st // tpb)
        beats[beat] += 1
    return beats
