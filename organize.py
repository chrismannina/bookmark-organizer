#!/usr/bin/env python
"""
Bookmark Organizer - AI-powered bookmark classification tool

This script organizes browser bookmarks by categorizing them using OpenAI's GPT models.
It takes an exported HTML bookmark file from your browser and creates a new HTML file
with bookmarks organized into a folder structure.

Dependencies:
- beautifulsoup4
- openai (v1.0.0+)
- python-dotenv
- tqdm
- scikit-learn

Setup:
1. Install dependencies: pip install beautifulsoup4 openai python-dotenv tqdm scikit-learn
2. Create a .env file with your OpenAI API key: OPENAI_API_KEY=your_key_here

Usage:
python organize.py input_bookmarks.html output_organized.html
"""

import os, re, html, argparse, datetime, sys, collections
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.cluster import KMeans

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # initialize client

# ---------- helpers ----------------------------------------------------------

SYSTEM_PROMPT = """You are an assistant that classifies browser bookmarks.
Return a concise UNIX-style folder path (<10 words total). 
Use high-level topics (Work, Education, AI, Home, Reading etc.) 
and nested sub-folders when useful. Only output the path."""

def parse_netscape(html_path: str):
    """Yield (title, url, add_date, original_folder)."""
    print(f"Reading bookmarks from {html_path}...")
    try:
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        
        # Simple check for expected bookmark format
        if 'HREF=' not in content and '<A HREF' not in content and '<a href' not in content:
            print("Warning: This doesn't appear to be a bookmark file (no HREF attributes found)")
        
        soup = BeautifulSoup(content, "html.parser")
        
        # Direct approach first - find all links
        all_links = soup.find_all('a')
        if all_links:
            print(f"Found {len(all_links)} links in the file")
            
            # Try to determine folders by parsing HTML structure
            extracted = []
            
            # First try the well-structured approach (Netscape format)
            if soup.find('dl'):
                extracted = list(extract_bookmarks(soup.find('dl')))
            
            # If that fails or returns incomplete results, use a more aggressive approach
            if len(extracted) < len(all_links) * 0.8:  # If we missed more than 20% of links
                print(f"Structured extraction found only {len(extracted)} of {len(all_links)} bookmarks. Enhancing extraction...")
                
                # Clear previous results if they're too incomplete
                if len(extracted) < len(all_links) * 0.5:  # If we missed more than half
                    extracted = []
                
                # Map of links we've already processed
                processed_urls = {item[1] for item in extracted}
                
                # Process each link directly with better folder detection
                for a in all_links:
                    url = a.get("href")
                    
                    # Skip if already processed or if it's not a proper URL
                    if not url or url in processed_urls or not (url.startswith('http') or url.startswith('https')):
                        continue
                    
                    title = a.get_text(strip=True)
                    add_date = a.get("add_date") or str(int(datetime.datetime.now().timestamp()))
                    
                    # Try multiple approaches to find the folder
                    folder = None
                    
                    # Approach 1: Look for parent H3 tag
                    parent_h3 = None
                    current = a
                    while current and current.parent:
                        h3 = current.find_previous_sibling('h3') or current.parent.find_previous_sibling('h3')
                        if h3:
                            parent_h3 = h3.get_text(strip=True)
                            break
                        current = current.parent
                    
                    # Approach 2: Check for DT->DL->H3 structure
                    if not parent_h3:
                        for parent in a.parents:
                            if parent.name == 'dt':
                                parent_dl = parent.parent
                                if parent_dl and parent_dl.name == 'dl':
                                    prev_sibling = parent_dl.find_previous_sibling()
                                    if prev_sibling and prev_sibling.name == 'h3':
                                        parent_h3 = prev_sibling.get_text(strip=True)
                                        break
                    
                    # Approach 3: Try to find any nearby H3 tags
                    if not parent_h3:
                        # Look at nearby headers within reasonable distance
                        current = a
                        for _ in range(5):  # Check up to 5 levels up
                            if current.parent:
                                current = current.parent
                                h3_tags = current.find_all('h3')
                                if h3_tags:
                                    # Find closest h3 tag
                                    closest_h3 = min(h3_tags, 
                                                    key=lambda h: abs(h.sourceline - a.sourceline) 
                                                    if hasattr(h, 'sourceline') and hasattr(a, 'sourceline') 
                                                    else float('inf'))
                                    parent_h3 = closest_h3.get_text(strip=True)
                                    break
                    
                    if parent_h3:
                        folder = parent_h3
                    
                    extracted.append((title, url, add_date, folder))
                    processed_urls.add(url)
            
            if extracted:
                print(f"Successfully extracted {len(extracted)} bookmarks")
                return extracted
            else:
                # Last resort - just extract basic link info with no folder structure
                print("Warning: Could not determine folder structure. Using basic link extraction.")
                return [(a.get_text(strip=True), 
                        a.get("href"), 
                        a.get("add_date") or str(int(datetime.datetime.now().timestamp())),
                        None) for a in all_links if a.get("href") and (a.get("href").startswith('http') or a.get("href").startswith('https'))]
        else:
            print("No links (<a> tags) found in the file.")
            
        # If we get here, try an alternative parsing approach
        print("Trying alternative parsing method...")
        
        # Some exports might use non-standard format, check for URLs
        url_pattern = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
        urls = url_pattern.findall(content)
        if urls:
            print(f"Found {len(urls)} URLs in the file using regex")
            # Create basic bookmarks with just the URL
            return [(url, url, str(int(datetime.datetime.now().timestamp())), None) 
                   for url in urls]
                
        return []
        
    except Exception as e:
        print(f"Error parsing bookmark file: {e}")
        import traceback
        traceback.print_exc()
        return []
    
def extract_bookmarks(element, folder_path=None):
    """Extract bookmarks from HTML with folder structure."""
    if element is None:
        return
        
    if folder_path is None:
        folder_path = []
    
    # If element is an H3 tag, it's a folder
    if element.name == 'h3':
        folder_name = element.get_text(strip=True)
        new_folder_path = folder_path + [folder_name]
        
        # Find the next DL after this H3, which contains items in this folder
        dl = element.find_next('dl')
        if dl:
            yield from extract_bookmarks(dl, new_folder_path)
    
    # If element is a DL tag, it contains items
    elif element and element.name == 'dl':
        for dt in element.find_all('dt', recursive=False):
            # If contains A tag, it's a bookmark
            a = dt.find('a')
            if a:
                yield (
                    a.get_text(strip=True),
                    a.get("href"),
                    a.get("add_date") or str(int(datetime.datetime.now().timestamp())),
                    "/".join(folder_path) if folder_path else None
                )
            
            # If contains H3 tag, it's a subfolder
            h3 = dt.find('h3')
            if h3:
                yield from extract_bookmarks(h3, folder_path)
            
            # If contains DL tag directly, process its content
            dl = dt.find('dl', recursive=False)
            if dl:
                yield from extract_bookmarks(dl, folder_path)

def analyze_bookmark_collection(bookmarks, max_sample=300):
    """Analyze the entire bookmark collection to determine optimal structure, focusing on user tasks."""
    print("Analyzing bookmark collection for user tasks and optimal folder structure...")

    # --- Existing analysis code (domains, folders, sample) ---
    original_folders = [folder for _, _, _, folder in bookmarks if folder]
    folder_counter = collections.Counter(original_folders)
    domains = {}
    for title, url, _, folder in bookmarks:
        # Normalize URL slightly for domain extraction
        url_norm = url.lower().replace("https://", "").replace("http://", "").replace("www.", "")
        try:
            domain = url_norm.split("/", 1)[0]
            domains.setdefault(domain, []).append((title, url, folder))
        except Exception:
            domains.setdefault("other", []).append((title, url, folder))

    domain_summary = []
    for domain, items in sorted(domains.items(), key=lambda x: len(x[1]), reverse=True):
        if len(items) > 1: # Show domains with at least 2 bookmarks
            titles = [t for t, _, _ in items[:5]]
            domain_summary.append(f"- {domain} ({len(items)} bookmarks)")
            # Only show sample titles for domains with > 3 bookmarks for brevity
            if len(items) > 3:
                 domain_summary.append(f"  Sample titles: {', '.join(titles[:3])}" + (f"... and {len(titles)-3} more" if len(titles) > 3 else ""))

    folder_summary = []
    for folder, count in folder_counter.most_common(10):
        if folder and count > 1:
            folder_summary.append(f"- {folder} ({count} bookmarks)")

    # Create a representative sample, avoiding duplicates
    bookmark_sample = []
    processed_for_sample = set()
    combined_list = bookmarks # Use the full list to ensure diverse sampling
    import random
    random.shuffle(combined_list) # Shuffle to get random sample across domains/folders

    for title, url, _, folder in combined_list:
         if len(bookmark_sample) >= max_sample:
             break
         # Use canonical URL for duplicate check if possible, otherwise original
         url_key = url.lower().replace("https://", "").replace("http://", "").replace("www.", "").split('?')[0].rstrip('/')
         if url_key not in processed_for_sample:
             bookmark_sample.append((title, url, folder))
             processed_for_sample.add(url_key)
             
    print(f"Created sample of {len(bookmark_sample)} unique bookmarks.")

    # --- LLM Step 1: Identify Personas and Tasks ---
    persona_system_prompt = """You are an expert information architect. Analyze the provided bookmark data (domains, original folders, sample titles)
and identify the primary user personas (e.g., Software Developer, Student, Home Cook, Financial Planner)
and the key tasks or goals this user is trying to achieve with these bookmarks (e.g., Learning Python, Managing Finances, Planning Travel, Finding Recipes).

Output format:
PERSONAS:
- [Persona 1]
- [Persona 2]
...
TASKS:
- [Task 1]
- [Task 2]
...
Be concise and base your analysis *only* on the provided data."""

    persona_user_prompt = f"""Analyze this bookmark collection:
Total bookmarks: {len(bookmarks)}
Most frequent domains (Top 10):
{chr(10).join(domain_summary[:10]) if domain_summary else "N/A"}

Original folder structure hints (Top 10):
{chr(10).join(folder_summary) if folder_summary else "N/A"}

Representative sample of bookmarks ({len(bookmark_sample)} items, showing up to 50):
{chr(10).join([f"- {title} (URL: {url}) {f'(from folder: {folder})' if folder else ''}" for title, url, folder in bookmark_sample[:50]])}

Based *only* on this data, identify the user's likely personas and key tasks. Respond ONLY in the specified format."""

    personas_and_tasks_text = "Personas: Unknown\\nTasks: Unknown" # Default value

    try:
        print("\\nStep 1: Identifying User Personas and Tasks...")
        persona_resp = client.chat.completions.create(
            model=args.model, # Use user-specified model for analysis
            messages=[
                {"role": "system", "content": persona_system_prompt},
                {"role": "user", "content": persona_user_prompt}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        personas_and_tasks_text = persona_resp.choices[0].message.content.strip()
        print("\\n===== PERSONA & TASK ANALYSIS =====")
        print(personas_and_tasks_text)
        print("===================================\\n")

    except Exception as e:
        print(f"Error during Persona/Task analysis: {e}")
        print("Proceeding without persona/task analysis.")

    # --- LLM Step 2: Generate Folder Structure based on Analysis ---
    structure_system_prompt = """You are an expert information architect designing a TASK-ORIENTED bookmark taxonomy.
Based on the user's personas, tasks, and bookmark samples, create a STREAMLINED folder structure.

RULES:
1.  **Baseline Categories:** Start with a standard set of top-level categories: `Communicate`, `Work`, `Learn`, `Finance`, `Home`, `Tech`, `Media`, `Productivity`, `Reference`, `Shopping`. Adapt/rename these slightly if the analysis strongly suggests it (e.g., `Learn` might become `Academics` for a student).
2.  **Add Specific Categories:** Add *additional* top-level categories ONLY if the user analysis clearly identifies major personas/tasks NOT covered by the baseline (e.g., a specific Hobby, Travel Planning).
3.  **Task-Focused Naming:** Prefer clear nouns or verbs reflecting user goals (e.g., "Learn Python", "Manage Finances").
4.  **Consolidated & Shallow:** Aim for 5-12 broad top-level categories maximum. Use subfolders (max depth 2, e.g., Category/Subcategory) sparingly, ONLY if a clear group of 4+ related bookmarks exists in the samples.
5.  **Consider Original:** Lightly consider the user's original folders as hints for naming or sub-folder creation, but prioritize the new task-oriented structure.
6.  **Mandatory Topics:** Ensure the final structure accommodates specific topics mentioned in the Persona/Task analysis (e.g., if "Pharmacy CE" was a task, ensure a `Health/Pharmacy CE` or similar path exists).
7.  **Output Format:** Return ONLY the folder structure as an indented list (using hyphens and spaces). Example:
    - Communicate
      - Email
    - Learn
      - Programming
      - Languages
    - Finance
    - Tech
      - AI & ML
    - Productivity
    - Reference
    - Home

DO NOT include explanations or assign individual bookmarks."""

    structure_user_prompt = f"""User analysis indicates the following:
{personas_and_tasks_text}

Original folder structure hints (top 10):
{chr(10).join(folder_summary) if folder_summary else "N/A"}

Representative sample of bookmarks ({len(bookmark_sample)} items, showing up to 50):
{chr(10).join([f"- {title} (URL: {url}) {f'(from folder: {folder})' if folder else ''}" for title, url, folder in bookmark_sample[:50]])}

Based on the user analysis, tasks, and bookmark samples, design a streamlined, task-oriented folder structure following ALL the rules (Baseline Categories, Consolidation, Depth, Naming, Mandatory Topics). Output ONLY the indented list structure."""

    structure = "" # Default value
    analysis_summary = personas_and_tasks_text # Use persona analysis as the main analysis output

    try:
        print("Step 2: Generating Task-Oriented Folder Structure...")
        structure_resp = client.chat.completions.create(
            model=args.model, # Use user-specified model
            messages=[
                {"role": "system", "content": structure_system_prompt},
                {"role": "user", "content": structure_user_prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        structure = structure_resp.choices[0].message.content.strip()
        print("\\n===== PROPOSED FOLDER STRUCTURE =====")
        print(structure)
        print("=====================================\\n")

    except Exception as e:
        print(f"Error during Structure Generation: {e}")
        print("Falling back to basic structure generation.")
        # Fallback: Use the old system prompt if the new one fails
        fallback_prompt = """You are an expert bookmark organizer... [Original Structure Prompt Here] ...""" # (Keep original prompt for brevity)
        # You might want to re-run the analysis call here with the simpler prompt.
        # For now, we'll return an empty structure on failure.
        structure = "- Uncategorized" # Simple fallback

    return analysis_summary, structure, folder_counter

def embed_and_cluster_bookmarks(bookmarks, max_clusters=20):
    """Create embeddings and cluster bookmarks by semantic similarity."""
    print(f"Generating embeddings for {len(bookmarks)} bookmarks...")
    titles_and_urls = [f"{title}: {url}" for title, url, _, _ in bookmarks]
    
    # Generate embeddings in batches if necessary (OpenAI API might have limits)
    batch_size = 1000  # Adjust as needed
    all_embeddings = []
    for i in tqdm(range(0, len(titles_and_urls), batch_size), desc="Generating embeddings"):
        batch = titles_and_urls[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small", # Using a cost-effective embedding model
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size}: {e}")
            # Handle error, e.g., add None placeholders or retry
            all_embeddings.extend([None] * len(batch)) # Placeholder for failed embeddings

    # Filter out failed embeddings and corresponding bookmarks
    valid_embeddings = [emb for emb in all_embeddings if emb is not None]
    valid_bookmarks = [bm for i, bm in enumerate(bookmarks) if all_embeddings[i] is not None]

    if not valid_embeddings:
        print("Error: No embeddings could be generated.")
        return {}, {} # Return empty dictionaries if embedding fails completely

    num_bookmarks_with_embeddings = len(valid_embeddings)
    print(f"Successfully generated embeddings for {num_bookmarks_with_embeddings} bookmarks.")

    # Determine the optimal number of clusters (e.g., min 5 bookmarks per cluster, up to max_clusters)
    n_clusters = min(max_clusters, num_bookmarks_with_embeddings // 5)
    if n_clusters < 1: n_clusters = 1 # Ensure at least one cluster
        
    print(f"Clustering {num_bookmarks_with_embeddings} bookmarks into {n_clusters} clusters...")
    
    # Cluster using K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init
    
    try:
      clusters = kmeans.fit_predict(valid_embeddings)
    except Exception as e:
        print(f"Error during clustering: {e}")
        # Fallback: Assign all to a single cluster
        clusters = [0] * num_bookmarks_with_embeddings
        n_clusters = 1


    # Group bookmarks by cluster_id
    clustered_bookmarks = collections.defaultdict(list)
    bookmark_to_cluster = {}
    for i, bookmark in enumerate(valid_bookmarks):
        cluster_id = clusters[i]
        clustered_bookmarks[cluster_id].append(bookmark)
        # Map original bookmark tuple (title, url) to cluster_id
        bookmark_to_cluster[(bookmark[0], bookmark[1])] = cluster_id
        
    print(f"Clustering complete. Found {len(clustered_bookmarks)} clusters.")
    
    # Add bookmarks with failed embeddings to a separate "unclustered" group if needed
    unclustered_bookmarks = [bm for i, bm in enumerate(bookmarks) if all_embeddings[i] is None]
    if unclustered_bookmarks:
      clustered_bookmarks[-1] = unclustered_bookmarks # Assign cluster_id -1 for unclustered
      for bm in unclustered_bookmarks:
          bookmark_to_cluster[(bm[0], bm[1])] = -1
      print(f"Assigned {len(unclustered_bookmarks)} bookmarks with failed embeddings to cluster -1.")

    return clustered_bookmarks, bookmark_to_cluster

def batch_classify(bookmarks, batch_size=10, collection_structure="", original_folders=None):
    """Classify bookmarks by clustering and assigning folders to clusters."""
    results = {}
    
    # 1. Embed and cluster bookmarks
    clustered_bookmarks, bookmark_to_cluster = embed_and_cluster_bookmarks(bookmarks)
    cluster_labels = {}
    
    # 2. Label each cluster using LLM
    print(f"Labeling {len(clustered_bookmarks)} clusters...")
    pbar_clusters = tqdm(clustered_bookmarks.items(), desc="Labeling clusters", unit="cluster")
    
    for cluster_id, cluster_items in pbar_clusters:
        pbar_clusters.set_description(f"Labeling cluster {cluster_id}")
        
        if cluster_id == -1: # Handle bookmarks with failed embeddings
            cluster_labels[cluster_id] = "Uncategorized" # Assign directly
            print(f"Cluster {cluster_id} (failed embeddings) assigned to Uncategorized.")
            continue
            
        if not cluster_items:
            print(f"Cluster {cluster_id} is empty, skipping.")
            continue

        # Take a sample of bookmarks from the cluster for labeling
        sample_size = min(len(cluster_items), 10) # Use up to 10 samples
        sample_content = []
        for j, (title, url, _, orig_folder) in enumerate(cluster_items[:sample_size]):
            folder_context = f" [Original folder: {orig_folder}]" if orig_folder else ""
            sample_content.append(f"Sample {j+1}:\nTitle: {title}\nURL: {url}{folder_context}")

        system_prompt = """You are an assistant that assigns a single folder path to a cluster of similar bookmarks.
Analyze the sample bookmarks provided from a cluster and determine the single best folder path from the provided structure.

FOCUS ON CONSOLIDATION:
1. Choose the broadest, most appropriate category from the STRUCTURE guide.
2. Avoid creating new subfolders unless absolutely necessary and justified by the samples.
3. Prioritize using the existing top-level categories.
4. If no category fits well, assign to 'Uncategorized'.

Return ONLY the single, concise folder path (e.g., Technology/AI, Finance, Education)."""
        
        if collection_structure:
            system_prompt += f"""

IMPORTANT: Use this folder structure as your guide:
{collection_structure}

Assign the cluster to ONE of these paths. DO NOT create new top-level categories.
"""
        else:
            system_prompt += "\n\nCreate a simple, high-level folder path (e.g., Technology, News, Shopping)."
            
        # Prepare the joined sample content outside the f-string
        joined_sample_content = "\n\n".join(sample_content)
            
        user_prompt = f"""Cluster contains {len(cluster_items)} bookmarks.
Here are {sample_size} samples from the cluster:

{joined_sample_content}

Please assign the most appropriate folder path for this entire cluster based on the provided structure guide.
Return ONLY the folder path.
"""

        try:
            resp = client.chat.completions.create(
                model=args.model,  # Use user-specified model (potentially more powerful)
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Low temperature for deterministic labeling
            )
            path = resp.choices[0].message.content.strip().lstrip("/")
            if not path:
                path = "Uncategorized" # Fallback if LLM returns empty string
            # Clean up potential LLM conversational additions
            if ":" in path:
                 path = path.split(":", 1)[-1].strip()
            if path.startswith("-"): path = path.lstrip("-").strip()
            if path.startswith("•"): path = path.lstrip("•").strip()
            if "folder path is:" in path.lower(): path = path.split(":")[-1].strip()
            # Ensure path doesn't exceed max depth (simple check for now)
            if path.count('/') > 2:
                path = "/".join(path.split('/')[:3])
                
            cluster_labels[cluster_id] = path if path else "Uncategorized"
            pbar_clusters.set_postfix(category=path if path else "Uncategorized")
            
        except Exception as e:
            print(f"Error labeling cluster {cluster_id}: {e}")
            cluster_labels[cluster_id] = "Uncategorized" # Fallback on error

    # 3. Assign the determined cluster label to all bookmarks in that cluster and apply heuristics
    print("Assigning cluster labels and applying heuristics...")
    for (title, url), cluster_id in bookmark_to_cluster.items():
        assigned_path = cluster_labels.get(cluster_id, "Uncategorized")
        
        # Heuristic Overrides:
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Email Clients
        if "mail.google.com" in url_lower or "outlook.live.com" in url_lower or "mail.protonmail.com" in url_lower:
            assigned_path = "Communicate/Email"
        # Social Media / Professional Networks
        elif "linkedin.com" in url_lower:
            assigned_path = "Career/LinkedIn" # Be more specific for LinkedIn
        elif "reddit.com" in url_lower:
             # Could be learn, tech, entertainment - check title/original path?
             if "/r/OSUOnlineCS" in url_lower or "/r/flask" in url_lower: # Example specific subreddits
                 assigned_path = "Learn/Tech Communities"
             elif "/r/datascience" in url_lower:
                  assigned_path = "Learn/Data Science"
             else:
                  assigned_path = "Media/Social Media/Reddit" # Default Reddit
        elif "twitch.tv" in url_lower or "youtube.com" in url_lower or "spotify.com" in url_lower or "hulu.com" in url_lower or "hbomax.com" in url_lower:
            # Check for educational YouTube links
            edu_keywords = ['tutorial', 'learn', 'lecture', 'course', 'cs', 'programming', 'python', 'javascript']
            if "youtube.com" in url_lower and any(keyword in title_lower for keyword in edu_keywords):
                assigned_path = "Learn/Videos & Tutorials"
            else:
                assigned_path = "Media/Entertainment"
        # Development Platforms
        elif "github.com" in url_lower:
             # Distinguish between learning repos and personal/professional?
             # Maybe check original folder? For now, general Dev
             assigned_path = "Tech/Development/GitHub"
        # Educational Platforms
        elif "canvas." in url_lower or ".edu" in url_lower.split('/')[0]: # Check domain
             if "oregonstate.edu" in url_lower:
                 assigned_path = "Learn/Academics/OSU"
             else:
                 assigned_path = "Learn/Academics"
        # Cloud Storage / Docs
        elif "drive.google.com" in url_lower or "docs.google.com" in url_lower or "onedrive.live.com" in url_lower:
             assigned_path = "Productivity/Cloud Storage & Docs"
        # Finance / Banking
        elif "schwab.com" in url_lower or "paypal.com" in url_lower or "fincen.gov" in url_lower or "monarchmoney.com" in url_lower:
             assigned_path = "Finance/Banking & Taxes"
             
        # Add more heuristics as needed...
        
        results[(title, url)] = assigned_path
        
    print("Batch classification complete.")
    return results

def classify(title: str, url: str) -> str:
    # This function is no longer directly used by batch_classify, 
    # but might be kept for potential single-item classification needs or debugging.
    # Consider removing if truly unused.
    """Return a category path from the LLM."""
    resp = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}, # Use the simpler original system prompt here
            {"role": "user",   "content": f"Title: {title}\nURL: {url}"}
        ],
        # temperature=0.2,
    )
    path = resp.choices[0].message.content.strip().lstrip("/")
    return path if path else "Uncategorized"

def build_tree(bookmarks):
    """Return nested dict {folder: subtree_or_list_of_links}."""
    tree = {}
    
    # Analyze the entire collection first
    bookmark_list = bookmarks
    analysis, structure, original_folders = analyze_bookmark_collection(bookmark_list)
    
    # Pre-classify all bookmarks using the new cluster-based method
    print("Classifying bookmarks based on clustering and analysis...")
    classifications = batch_classify(bookmark_list, collection_structure=structure, original_folders=original_folders)
    
    # Ensure we have at least one classification for each bookmark
    for title, url, add_date, _ in bookmark_list:
        if (title, url) not in classifications or not classifications[(title, url)]:
            # Assign default category if missing
            print(f"Warning: Bookmark '{(title, url)}' was not classified. Assigning to Uncategorized.")
            classifications[(title, url)] = "Uncategorized"
    
    # Count categories to confirm they're being used
    category_counts = collections.Counter([path.split('/')[0] for path in classifications.values()])
    print("\nCategory distribution after classification:")
    for category, count in category_counts.most_common():
        print(f"- {category}: {count} bookmarks")
    
    # Build tree with the classifications
    pbar = tqdm(bookmark_list, desc="Building folder structure", unit="bookmark")
    for title, url, add_date, _ in pbar:
        pbar.set_description(f"Processing: {title[:40]}{'...' if len(title) > 40 else ''}")
        try:
            # Get path, ensure it exists
            path_str = classifications.get((title, url))
            if not path_str or path_str.strip() == "":
                path_str = "Uncategorized"
                
            # Split path into folder hierarchy
            path = path_str.split("/")
            pbar.set_postfix(category=f"{'/' + '/'.join(path) if path else 'Uncategorized'}") # Added prefix slash for clarity
            
            # Create folder hierarchy
            node = tree
            for part in path:
                if not part:  # Skip empty path components
                    continue
                # Ensure keys are strings
                part = str(part).strip()
                if not part: continue # Skip if part becomes empty after stripping
                node = node.setdefault(part, {})
            
            # Add link to the leaf folder
            if isinstance(node, dict):
              node.setdefault("_links", []).append((title, url, add_date))
            else:
              # This case might happen if a path part conflicts with _links, handle gracefully
              print(f"Warning: Could not add link '{title}' to node {path}. Node is not a dictionary.")
              parent_node = tree # Find parent node to add to uncategorized or similar
              # For simplicity, add to root Uncategorized for now
              uncategorized_node = tree.setdefault("Uncategorized", {})
              uncategorized_node.setdefault("_links", []).append((title, url, add_date))

        except Exception as e:
            pbar.write(f"Error processing {title} ({url}): {e}")
            import traceback
            pbar.write(traceback.format_exc())
            # Ensure the bookmark is still added somewhere
            node = tree.setdefault("Uncategorized", {})
            node.setdefault("_links", []).append((title, url, add_date))
    
    # Create default folders for empty categories to ensure structure matches analysis
    if not tree and len(bookmark_list) > 0:
        print("Warning: No folders were created. Creating default structure...")
        # Extract top-level categories from the analysis structure
        if structure:
            # Try to parse main categories from the structure description
            categories = []
            lines = structure.split('\n')
            for line in lines:
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    # Extract category name
                    category = line.strip().lstrip('-').lstrip('•').strip()
                    if '/' in category:
                        category = category.split('/')[0].strip()
                    if '(' in category:
                        category = category.split('(')[0].strip()
                    if category:
                        categories.append(category)
            
            # Create empty folders for each category
            for category in categories:
                if category not in tree:
                    tree[category] = {"_links": []}
    
    # Consolidate the tree to eliminate folders with only a single bookmark or subfolder
    if not args.skip_consolidation and tree:
        tree = consolidate_tree(tree)
    
    return tree, analysis

def consolidate_tree(node, parent_path=""):
    """Eliminate excessive nesting and folders with just 1-2 items."""
    if not isinstance(node, dict):
        return node
    
    # Process children first (bottom-up approach)
    for name, child in list(node.items()):
        if name != "_links" and isinstance(child, dict):
            node[name] = consolidate_tree(child, f"{parent_path}/{name}" if parent_path else name)
    
    # Check if this node has only one subfolder and no direct links
    if len(node) == 1 and "_links" not in node:
        # Get the only child's name and content
        child_name, child_content = next(iter(node.items()))
        
        # If child has direct links but no other folders, merge with parent
        if "_links" in child_content and len(child_content) == 1:
            return {"_links": child_content["_links"]}
        
        # If child has only a few links and no other folders, merge with parent
        if "_links" in child_content and len(child_content) == 2 and len(child_content["_links"]) <= 2:
            links = child_content["_links"]
            other_child_name, other_child_content = next((k, v) for k, v in child_content.items() if k != "_links")
            # Only merge if the *other* child is also very small (<=1 link)
            if len(other_child_content.get("_links", [])) <= 1 and len(other_child_content) == 1:
                # Merge both child's links and grandchild's links into parent
                all_links = links + other_child_content["_links"]
                return {"_links": all_links}
    
    # Check for small folders that could be consolidated
    small_folders = {}
    for name, child in list(node.items()):
        if name != "_links" and isinstance(child, dict):
            # Check if it's a small folder (1 link and no subfolders, or 0 links and 1 small subfolder)
            num_links = len(child.get("_links", []))
            subfolder_count = sum(1 for k, v in child.items() if k != "_links" and isinstance(v, dict))
            only_small_subfolders = all(k == "_links" or (isinstance(v, dict) and len(v.get("_links", [])) <= 1 and len(v) == 1) 
                                     for k, v in child.items() if k != "_links")
                                     
            # Condition: Folder has 1 link and no subfolders, OR 0 links and only 1 small subfolder
            if (num_links == 1 and subfolder_count == 0) or \
               (num_links == 0 and subfolder_count == 1 and only_small_subfolders):
                # Let's not automatically mark for consolidation here, handle single-item folders below
                pass 
            # Condition: Folder has <= 2 links AND only contains small subfolders (or no subfolders)
            elif num_links <= 2 and only_small_subfolders and subfolder_count <= 1: # Relaxed condition
                 small_folders[name] = child

    # If we have multiple *very* small folders (e.g., <=2 items total each), consider merging them
    # This logic might need careful tuning based on desired outcome
    folders_to_merge = {}
    for name, folder in small_folders.items():
        total_items = len(folder.get("_links", [])) + \
                      sum(len(subfolder.get("_links", [])) for subname, subfolder in folder.items() if subname != "_links")
        if total_items <= 2: # Only consider merging if the total content is tiny
            folders_to_merge[name] = folder

    if len(folders_to_merge) > 1 and len(folders_to_merge) <= 3: # Keep condition to merge 2 or 3
        # Create a new merged folder (use the parent path + a generic name? or first name?)
        # Using first name for simplicity, but could be improved
        merged_name = next(iter(folders_to_merge.keys())) 
        merged_folder = {"_links": []}
        print(f"Consolidating {list(folders_to_merge.keys())} into {merged_name} at path {parent_path}")
        
        # Collect all links
        for name, folder in folders_to_merge.items():
            if "_links" in folder:
                merged_folder["_links"].extend(folder["_links"])
            for subname, subfolder in folder.items():
                if subname != "_links" and isinstance(subfolder, dict) and "_links" in subfolder:
                    merged_folder["_links"].extend(subfolder["_links"])
        
        # Remove the small folders and add the merged one
        if merged_folder["_links"]: # Only add if there are actually links
          for name in folders_to_merge:
              if name in node: del node[name]
          node[merged_name] = merged_folder
        else: # If no links, just remove the empty structures
           print(f"Skipping consolidation for {merged_name} as it resulted in no links.")
           for name in folders_to_merge:
              if name in node: del node[name] 

    # Final check: Remove completely empty folders that might result from consolidation
    for name, child in list(node.items()):
        if name != "_links" and isinstance(child, dict) and not child:
            print(f"Removing empty folder: {parent_path}/{name}")
            del node[name]
            
    return node

def display_tree(node, prefix="", is_last=True, max_links=3):
    """Print tree structure of bookmarks organization."""
    for i, (name, child) in enumerate(sorted(node.items())):
        is_last_item = i == len(node) - 1
        
        if name == "_links":
            num_links = len(child)
            print(f"{prefix}{'└── ' if is_last else '├── '}📑 {num_links} bookmark{'s' if num_links != 1 else ''}")
            
            # Show sample of bookmarks in this folder
            for j, (title, _, _) in enumerate(child[:max_links]):
                if j < max_links:
                    print(f"{prefix}{'    ' if is_last else '│   '}{'└── ' if j == min(max_links-1, num_links-1) else '├── '}{title[:60]}")
                
            # Show if there are more we're not displaying
            if num_links > max_links:
                print(f"{prefix}{'    ' if is_last else '│   '}    ... and {num_links - max_links} more")
        else:
            # Folder
            print(f"{prefix}{'└── ' if is_last else '├── '}📁 {name}")
            
            # Recursively display children with updated prefix
            new_prefix = prefix + ('    ' if is_last else '│   ')
            display_tree(child, new_prefix, is_last_item)

def analyze_tree_structure(node, stats=None, path=None):
    """Analyze the tree structure for stats like folder distribution."""
    if stats is None:
        stats = {"total_folders": 0, "total_bookmarks": 0, "folder_sizes": [], "folder_depth": [], "single_bookmark_folders": 0}
    if path is None:
        path = []
    
    # Count this folder
    if path:  # Don't count the root
        stats["total_folders"] += 1
        stats["folder_depth"].append(len(path))
    
    bookmark_count = 0
    if "_links" in node:
        bookmark_count = len(node["_links"])
        stats["total_bookmarks"] += bookmark_count
        
        if bookmark_count > 0:
            stats["folder_sizes"].append(bookmark_count)
            
        # Count folders with only one bookmark and no subfolders
        if bookmark_count == 1 and len(node) == 1:
            stats["single_bookmark_folders"] += 1
    
    # Recurse into subfolders
    for name, child in node.items():
        if name != "_links" and isinstance(child, dict):
            analyze_tree_structure(child, stats, path + [name])
    
    return stats

def edit_tree_interactive(tree):
    """Allow users to interactively edit the tree structure."""
    while True:
        print("\nOptions:")
        print("1. Continue with current organization")
        print("2. View structure statistics")
        print("3. Cancel and exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            return tree
        elif choice == '2':
            stats = analyze_tree_structure(tree)
            print("\n=== Bookmark Organization Statistics ===")
            print(f"Total folders: {stats['total_folders']}")
            print(f"Total bookmarks: {stats['total_bookmarks']}")
            
            if stats['total_folders'] > 0:
                print(f"Single-bookmark folders: {stats['single_bookmark_folders']} ({stats['single_bookmark_folders']/stats['total_folders']*100:.1f}% of folders)")
                
                sizes = stats["folder_sizes"]
                if sizes:
                    print(f"Average bookmarks per folder: {sum(sizes)/len(sizes):.1f}")
                    print(f"Folder size distribution: min={min(sizes)}, max={max(sizes)}")
                    
                    size_groups = {
                        "1 bookmark": len([s for s in sizes if s == 1]),
                        "2-3 bookmarks": len([s for s in sizes if 2 <= s <= 3]),
                        "4-10 bookmarks": len([s for s in sizes if 4 <= s <= 10]),
                        "11+ bookmarks": len([s for s in sizes if s > 10])
                    }
                    for group, count in size_groups.items():
                        print(f"  {group}: {count} folders ({count/len(sizes)*100:.1f}%)")
                
                depths = stats["folder_depth"]
                if depths:
                    print(f"Folder depth distribution: min={min(depths)}, max={max(depths)}, avg={sum(depths)/len(depths):.1f}")
            else:
                print("No folders were created - there may be no bookmarks in the input file.")
            
            continue
        elif choice == '3':
            print("Operation cancelled.")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

def write_dl(node, indent=4):
    """Recursively emit DL/DT HTML for a node dict."""
    pad = " " * indent
    for name, child in sorted(node.items()):
        if name == "_links":  # plain links
            for title, url, add_date in child:
                yield (f'{pad}<DT><A HREF="{html.escape(url)}" '
                       f'ADD_DATE="{add_date}">{html.escape(title)}</A>\n')
        else:                 # sub-folder
            yield f'{pad}<DT><H3>{html.escape(name)}</H3>\n'
            yield f'{pad}<DL><p>\n'
            yield from write_dl(child, indent + 4)
            yield f'{pad}</DL><p>\n'

def export_netscape(tree, out_path):
    print(f"Exporting organized bookmarks to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE NETSCAPE-Bookmark-file-1>\n<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">\n<TITLE>Bookmarks</TITLE>\n<H1>Bookmarks</H1>\n<DL><p>\n""")
        lines = list(write_dl(tree, 4))
        for line in tqdm(lines, desc="Writing HTML", unit="line"):
            f.write(line)
        f.write("</DL><p>\n")

def verify_bookmarks(input_path, output_path):
    """Verify that all input URLs exist in the output file."""
    print("Verifying bookmark preservation...")
    
    # Extract URLs from input file
    input_soup = BeautifulSoup(open(input_path, encoding="utf-8"), "html.parser")
    input_urls = set(a.get("href") for a in input_soup.find_all("a"))
    
    # Extract URLs from output file
    output_soup = BeautifulSoup(open(output_path, encoding="utf-8"), "html.parser")
    output_urls = set(a.get("href") for a in output_soup.find_all("a"))
    
    # Check if all input URLs are in output
    missing_urls = input_urls - output_urls
    
    if not missing_urls:
        print(f"✓ All {len(input_urls)} bookmarks were preserved in the output file.")
        return True
    else:
        print(f"⚠ WARNING: {len(missing_urls)} bookmarks were not preserved in the output!")
        print(f"  Input: {len(input_urls)} URLs, Output: {len(output_urls)} URLs")
        if len(missing_urls) <= 5:
            for url in missing_urls:
                print(f"  Missing: {url}")
        return False

# ---------- CLI --------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI bookmark organiser")
    ap.add_argument("infile",  help="exported HTML from browser")
    ap.add_argument("outfile", help="new HTML to import")
    ap.add_argument("--auto", action="store_true", help="Skip interactive confirmation")
    ap.add_argument("--skip-consolidation", action="store_true", help="Skip folder consolidation step")
    ap.add_argument("--model", default="gpt-4o", choices=["gpt-4o", "gpt-4o-mini"], 
                    help="OpenAI model to use (default: gpt-4o") # o4-mini a chain of thought model, so you cannot use temperature or max_tokens
    ap.add_argument("--debug", action="store_true", help="Print detailed debugging information")
    args = ap.parse_args()

    # If in debug mode, print file structure
    if args.debug:
        print("DEBUG MODE: Analyzing bookmark file structure...")
        try:
            with open(args.infile, 'r', encoding='utf-8') as f:
                content = f.read()
                
            soup = BeautifulSoup(content, 'html.parser')
            
            # Print basic file stats
            print(f"File size: {len(content)} bytes")
            print(f"Total <a> tags: {len(soup.find_all('a'))}")
            print(f"Total <dl> tags: {len(soup.find_all('dl'))}")
            print(f"Total <h3> tags: {len(soup.find_all('h3'))}")
            
            # Show first few links if any
            links = soup.find_all('a', limit=3)
            if links:
                print("\nSample links found:")
                for i, link in enumerate(links):
                    print(f"Link {i+1}: {link.get('href')} - '{link.get_text(strip=True)}'")
            else:
                print("\nNo links found in file")
                
            # Check if this looks like a Netscape bookmark file
            if soup.find('META', attrs={'HTTP-EQUIV': 'Content-Type'}):
                print("File appears to have Netscape bookmark format headers")
            else:
                print("File does not have standard Netscape bookmark format headers")
                
        except Exception as e:
            print(f"Error analyzing file structure: {e}")
    
    parsed = parse_netscape(args.infile)
    
    # Deduplicate bookmarks based on URL
    print(f"\nFound {len(parsed)} raw bookmarks. Deduplicating...")
    unique_bookmarks_map = {}
    duplicates_count = 0
    for title, url, add_date, folder in parsed:
        # Normalize URL for deduplication (simple version: lowercase, strip trailing /)
        norm_url = url.lower().rstrip('/')
        if norm_url not in unique_bookmarks_map:
            unique_bookmarks_map[norm_url] = (title, url, add_date, folder)
        else:
            duplicates_count += 1
            # Optional: could implement logic to keep the one with the earliest add_date or longest title
            
    bookmarks = list(unique_bookmarks_map.values())
    print(f"Removed {duplicates_count} duplicate URLs. Processing {len(bookmarks)} unique bookmarks.")

    # Check if any bookmarks were found
    if not bookmarks: # Check the deduplicated list
        print("\n⚠️ No bookmarks found in the input file. Please check that it contains valid bookmarks.")
        print("Try running with --debug flag for more information")
        print("Exiting without creating output file.")
        sys.exit(1)
        
    tree, analysis = build_tree(bookmarks)
    
    # Display tree structure for user review
    print("\n=== Bookmark Collection Analysis ===")
    print(analysis)
    
    print("\n=== Proposed Bookmark Organization ===")
    display_tree(tree)
    
    # Display structure statistics
    stats = analyze_tree_structure(tree)
    print("\n=== Organization Statistics ===")
    print(f"Total folders: {stats['total_folders']}")
    if stats['total_folders'] > 0:
        print(f"Single-bookmark folders: {stats['single_bookmark_folders']} ({stats['single_bookmark_folders']/stats['total_folders']*100:.1f}% of folders)")
        if stats["folder_sizes"]:
            print(f"Average bookmarks per folder: {sum(stats['folder_sizes'])/len(stats['folder_sizes']):.1f}")
    else:
        print("No folders were created - there may be no bookmarks in the input file.")
    print("\nThis is how your bookmarks will be organized.")
    
    # Allow user to confirm or modify unless auto mode is enabled
    if not args.auto:
        tree = edit_tree_interactive(tree)
    
    export_netscape(tree, args.outfile)
    verify_bookmarks(args.infile, args.outfile)
    print(f"Complete! Organized bookmarks written to {args.outfile}")
