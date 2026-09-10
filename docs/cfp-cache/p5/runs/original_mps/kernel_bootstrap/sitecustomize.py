import sys
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_mtlearn_editable']
sys.path[:0] = ['/Volumes/SSD/GitHub/mtlearn/mtlearn/python', '/Volumes/SSD/GitHub/mtlearn/build/cfp-cache-p0-release/mtlearn/bindings']
