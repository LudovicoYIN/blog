#!/usr/bin/env python3
"""
批量更新博客图片引用，使用 PaddleOCR 重新处理后的新文件名。
"""
import os
import re
import glob

# Paper ID to available images mapping
PAPER_IMAGES = {}

def get_paper_images():
    """Get all available images for each paper."""
    papers_dir = "/home/llm/blog/public/papers"
    for paper_dir in os.listdir(papers_dir):
        paper_path = os.path.join(papers_dir, paper_dir)
        if os.path.isdir(paper_path):
            images = []
            for root, dirs, files in os.walk(paper_path):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        images.append(file)
            if images:
                PAPER_IMAGES[paper_dir] = images
    return PAPER_IMAGES

def update_post(post_path: str):
    """Update image references in a blog post."""
    with open(post_path, 'r') as f:
        content = f.read()
    
    updated = False
    
    # Find all image references
    image_pattern = r'(!\[[^\]]*\]\()([^)]+)(\)|<img[^>]*src="([^"]+)")'
    
    def replace_image(match):
        nonlocal updated
        full_match = match.group(0)
        image_path = match.group(2) or match.group(4)
        
        if not image_path:
            return full_match
        
        # Extract paper ID from path
        paper_match = re.search(r'/papers/(\d+\.\d+)', image_path)
        if not paper_match:
            return full_match
        
        paper_id = paper_match.group(1)
        old_filename = os.path.basename(image_path)
        
        if paper_id in PAPER_IMAGES:
            # Find the best matching image
            new_filename = PAPER_IMAGES[paper_id][0]  # Simple fallback
            new_path = f"/blog/papers/{paper_id}/{new_filename}"
            
            alt_match = re.search(r'!\[([^\]]*)\]', full_match)
            if alt_match:
                alt_text = alt_match.group(1)
                updated = True
                return f"![{alt_text}]({new_path})"
            elif '<img' in full_match:
                updated = True
                return f'<img src="{new_path}">'
        
        return full_match
    
    new_content = re.sub(image_pattern, replace_image, content)
    
    if updated:
        with open(post_path, 'w') as f:
            f.write(new_content)
        print(f"Updated: {post_path}")

def main():
    print("Scanning paper images...")
    get_paper_images()
    
    for paper_id, images in PAPER_IMAGES.items():
        print(f"  {paper_id}: {len(images)} images")
    
    print("\nUpdating blog posts...")
    posts_dir = "/home/llm/blog/src/content/posts"
    for post_file in glob.glob(os.path.join(posts_dir, "*.md")):
        update_post(post_file)

if __name__ == "__main__":
    main()
