import re

def process_coords(raw):
    matches = re.findall(r'\(([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\)', raw)
    coords = [[float(x), float(y)] for x, y in matches]
    modified = re.sub(r'\(([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\)', '<|loc|>', raw)
    return coords, modified

def process_desc(desc):
    banned = ['rear', 'Behind', 'behind', 'right-rear', 'left-rear']
    sentences = desc.split('. ')
    filtered = []
    for s in sentences:
        f = True
        for b in banned:
            if b in s:
                f = False
                break
        if f:
            filtered.append(s)
    new_desc = ''
    for s in filtered:
        new_desc += s
        new_desc += '. '
    new_desc = new_desc[:-2]
    return new_desc

def check_range(rmin, rmax, coords):
    for c in coords:
        for x in c:
            if not rmin < x < rmax:
                return False
    return True