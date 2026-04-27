"""Create a jieba_fast shim that redirects to jieba."""
import os
import site

sp = [p for p in site.getsitepackages() if 'site-packages' in p][0]
print(f'Site packages: {sp}')

jieba_fast_dir = os.path.join(sp, 'jieba_fast')
os.makedirs(jieba_fast_dir, exist_ok=True)

# Write __init__.py
init_content = (
    "# Shim: redirect jieba_fast to jieba\n"
    "from jieba import *\n"
    "import jieba as _jieba\n"
    "import sys\n"
    "sys.modules['jieba_fast'] = _jieba\n"
)

init_path = os.path.join(jieba_fast_dir, '__init__.py')
with open(init_path, 'w') as f:
    f.write(init_content)
print(f'Created jieba_fast shim at {init_path}')

# Also create _compat.py  
compat_path = os.path.join(jieba_fast_dir, '_compat.py')
with open(compat_path, 'w') as f:
    f.write('# Compatibility shim\n')
print(f'Created _compat shim at {compat_path}')

# Test import
import importlib
import jieba_fast
print(f'jieba_fast imported successfully: {jieba_fast}')
