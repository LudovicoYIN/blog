#!/usr/bin/env python3
"""Fix duplicate image references by distributing PaddleOCR images across blog post figure refs."""
import glob, re, os

# For each blog post, distribute available images across image references
fixed = 0
for f in sorted(glob.glob('src/content/posts/*.md')):
    with open(f) as fh:
        content = fh.read()
    
    # Find paper ID from image refs
    refs_match = re.findall(r'/blog/papers/(\d+\.\d+)/', content)
    if not refs_match:
        continue
    pid = refs_match[0]
    
    # Get all image refs in this file with positions
    img_refs = list(re.finditer(r'!\[(.*?)\]\(/blog/papers/' + re.escape(pid) + r'/([^)]+)\)', content))
    if not img_refs:
        continue
    
    # Get available images, sorted by bounding box position (top-left first)
    img_dir = f'public/papers/{pid}'
    available = sorted(glob.glob(f'{img_dir}/*.jpg'))
    
    unique_refs = list(set(m.group(2) for m in img_refs))
    
    if len(unique_refs) == len(img_refs):
        # All refs are already unique
        continue
    
    if not available:
        # No images available (e.g., adapld uses page6.png which doesn't exist)
        print(f"SKIP {pid}: no images available")
        continue
    
    # Distribute images: each ref gets a different image, cycling if needed
    print(f"FIX {pid} ({os.path.basename(f)}): {len(img_refs)} refs -> {len(available)} images")
    
    for idx, m in enumerate(img_refs):
        alt_text = m.group(1)
        old_fn = m.group(2)
        new_fn = os.path.basename(available[idx % len(available)])
        
        if old_fn != new_fn:
            old_ref = m.group(0)
            new_ref = f'![{alt_text}](/blog/papers/{pid}/{new_fn})'
            content = content.replace(old_ref, new_ref, 1)
            print(f"  [{alt_text}]: {old_fn} -> {new_fn}")
    
    with open(f, 'w') as fh:
        fh.write(content)
    fixed += 1

print(f"\nFixed {fixed} files")
