import sys, re
src = open('train_gpt2.py').read()
name = sys.argv[1]
# overrides as key=val pairs
ov = dict(a.split('=') for a in sys.argv[2:])
def setline(src, field, val):
    pat = re.compile(rf'(\n\s*{field} : [^=]+= )([^\n#]+)(#.*)?')
    def repl(m):
        comment = m.group(3) or ''
        return f'{m.group(1)}{val} {comment}'
    new, n = pat.subn(repl, src, count=1)
    assert n==1, f'field {field} not found/uniq ({n})'
    return new
for k,v in ov.items():
    src = setline(src, k, v)
open(f'exp/{name}.py','w').write(src)
print(f'wrote exp/{name}.py with {ov}')
