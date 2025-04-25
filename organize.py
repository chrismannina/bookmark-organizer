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
- requests

Setup:
1. Install dependencies: pip install beautifulsoup4 openai python-dotenv tqdm scikit-learn requests
2. Create a .env file with your OpenAI API key: OPENAI_API_KEY=your_key_here

Usage:
python organize.py input_bookmarks.html output_organized.html
"""

import os, re, html, argparse, datetime, sys, collections, requests
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
    
def extract_bookmarks(element, folder_path=None, is_toolbar=False):
    """Extract bookmarks from HTML with folder structure, noting toolbar."""
    if element is None:
        return
        
    if folder_path is None:
        folder_path = []
    
    # Check if the current H3 indicates the toolbar folder
    current_is_toolbar = is_toolbar
    if element.name == 'h3':
        folder_name = element.get_text(strip=True)
        # Check for common toolbar names or the specific attribute
        if element.get('personal_toolbar_folder', 'false').lower() == 'true' or \
           folder_name.lower() in ['favorites bar', 'bookmarks bar', 'toolbar']:
           current_is_toolbar = True
           folder_path = ['Toolbar'] # Standardize toolbar path
        else:
           current_is_toolbar = False # Reset if nested folder isn't toolbar
           folder_path = folder_path + [folder_name]
        
        # Find the next DL after this H3, which contains items in this folder
        dl = element.find_next('dl')
        if dl:
            yield from extract_bookmarks(dl, folder_path, current_is_toolbar)
    
    # If element is a DL tag, it contains items
    elif element and element.name == 'dl':
         # Check if THIS DL directly follows a toolbar H3
        prev_dt = element.find_previous('dt')
        if prev_dt:
             h3_in_prev_dt = prev_dt.find('h3')
             if h3_in_prev_dt and h3_in_prev_dt.get('personal_toolbar_folder', 'false').lower() == 'true':
                 current_is_toolbar = True
                 folder_path = ['Toolbar'] # Set path for items directly in toolbar DL

        for dt in element.find_all('dt', recursive=False):
            # If contains A tag, it's a bookmark
            a = dt.find('a')
            if a:
                 # Use the potentially updated folder path
                 current_folder_name = "/".join(folder_path) if folder_path else None
                 # If it's identified as toolbar, ensure path reflects that
                 if current_is_toolbar:
                      current_folder_name = "Toolbar"
                 yield (
                    a.get_text(strip=True),
                    a.get("href"),
                    a.get("add_date") or str(int(datetime.datetime.now().timestamp())),
                    current_folder_name
                )
            
            # If contains H3 tag, it's a subfolder - pass toolbar status down
            h3 = dt.find('h3')
            if h3:
                yield from extract_bookmarks(h3, folder_path, current_is_toolbar)
            
            # If contains DL tag directly, process its content - pass toolbar status down
            dl = dt.find('dl', recursive=False)
            if dl:
                yield from extract_bookmarks(dl, folder_path, current_is_toolbar)

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

def embed_and_cluster_bookmarks(bookmarks, args, max_clusters=20):
    """Create embeddings and cluster bookmarks by semantic similarity, optionally fetching content."""
    print(f"Preparing data for {len(bookmarks)} bookmarks...")

    content_for_embedding = []
    # Use a session for potential connection pooling
    session = requests.Session()
    session.headers.update({'User-Agent': 'BookmarkOrganizer/1.0 (+https://github.com/user/repo)'}) # Identify the bot

    fetch_pbar = tqdm(bookmarks, desc="Fetching content", unit="bookmark", disable=not args.fetch_content)
    for title, url, _, _ in fetch_pbar:
        text_to_embed = f"{title}: {url}" # Default text

        if args.fetch_content:
            fetch_pbar.set_description(f"Fetching {url[:50]}...")
            try:
                response = session.get(url, timeout=5, allow_redirects=True) # Added timeout and redirects
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                # Check content type - only parse HTML
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    fetched_title = soup.find('title')
                    fetched_title_text = fetched_title.get_text(strip=True) if fetched_title else title # Fallback to original title

                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    meta_desc_text = meta_desc['content'].strip() if meta_desc and meta_desc.get('content') else ""

                    # Combine title, description, and URL for richer context
                    text_to_embed = f"Title: {fetched_title_text}\nDescription: {meta_desc_text}\nURL: {url}"
                    if args.debug: print(f"\nFetched data for {url}:\n{text_to_embed}\n---")

                else:
                    if args.debug: print(f"\nSkipping non-HTML content ({content_type}) for {url}")
                    # Keep default text_to_embed (title: url) for non-HTML

            except requests.exceptions.Timeout:
                 if args.debug: print(f"\nTimeout fetching {url}")
            except requests.exceptions.RequestException as e:
                 if args.debug: print(f"\nError fetching {url}: {e}")
            except Exception as e: # Catch other potential parsing errors
                if args.debug: print(f"\nError processing {url}: {e}")
                
        content_for_embedding.append(text_to_embed)

    print(f"Generating embeddings for {len(content_for_embedding)} items...")
    # Generate embeddings in batches if necessary (OpenAI API might have limits)
    batch_size = 1000  # Adjust as needed
    all_embeddings = []
    # Use content_for_embedding instead of titles_and_urls
    embed_pbar = tqdm(range(0, len(content_for_embedding), batch_size), desc="Generating embeddings")
    for i in embed_pbar:
        batch = content_for_embedding[i:i+batch_size]
        embed_pbar.set_description(f"Embedding batch {i//batch_size + 1}")
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
    # Ensure valid_bookmarks corresponds to the items that got embeddings
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

def classify_bookmarks(bookmarks, collection_structure):
    """Classify each bookmark individually against the proposed structure."""
    print(f"Performing initial classification for {len(bookmarks)} bookmarks...")
    results = {}
    batch_size = 20 # Experiment with batch size for API calls
    batches = [bookmarks[i:i+batch_size] for i in range(0, len(bookmarks), batch_size)]

    system_prompt = f"""You are an assistant classifying bookmarks into a predefined folder structure.
Assign EACH bookmark to the *most appropriate* path from the structure provided below.
The structure uses '/' for hierarchy (e.g., Tech/AI).

RULES:
1.  **Strict Adherence:** ONLY use folder paths that exist in or can be directly derived from the provided structure.
2.  **Best Fit:** Choose the most specific relevant path. If a bookmark fits 'Tech/AI', use that instead of just 'Tech'.
3.  **Top-Level OK:** If no sub-category fits well, assign to the top-level category (e.g., 'Tech').
4.  **Uncategorized:** If a bookmark clearly does not fit *any* category in the structure, assign it the path 'Uncategorized'. Do not invent new categories.
5.  **Format:** Respond ONLY with a numbered list matching the input bookmarks, followed by the assigned path. Example:
    1. Tech/Development
    2. Communicate/Email
    3. Uncategorized

PROVIDED STRUCTURE:
{collection_structure}
"""

    for batch in tqdm(batches, desc="Classifying bookmarks", unit="batch"):
        batch_content = []
        for j, (title, url, _, orig_folder) in enumerate(batch):
            folder_context = f" [Original folder: {orig_folder}]" if orig_folder else ""
            batch_content.append(f"Bookmark {j+1}:\\nTitle: {title}\\nURL: {url}{folder_context}")

        # Pre-join the content to avoid backslash in f-string expression
        joined_batch_content = "\\n\\n".join(batch_content)

        user_prompt = f"""Please classify the following {len(batch)} bookmarks based *only* on the structure provided in the system prompt.

{joined_batch_content}

Respond ONLY with the numbered list and assigned paths."""

        try:
            resp = client.chat.completions.create(
                model=args.model, # Use the main model for initial classification
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
            )
            classifications_text = resp.choices[0].message.content.strip()
            
            # Parse response
            lines = classifications_text.split('\\n')
            for j, (title, url, _, _) in enumerate(batch):
                found = False
                for line in lines:
                    line = line.strip()
                    if line.startswith(f"{j+1}."):
                        parts = line.split(".", 1)
                        if len(parts) > 1:
                            path = parts[1].strip().lstrip("/")
                            if path:
                                results[(title, url)] = path
                                found = True
                                break
                if not found:
                    results[(title, url)] = "Uncategorized" # Fallback if parsing fails for an item

        except Exception as e:
            print(f"Error classifying batch: {e}. Assigning batch to Uncategorized.")
            for title, url, _, _ in batch:
                results[(title, url)] = "Uncategorized"

    # --- Apply Heuristics ---
    print("Applying classification heuristics...")
    refined_results = {}
    for (title, url), assigned_path in results.items():
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Email Clients
        if "mail.google.com" in url_lower or "outlook.live.com" in url_lower or "mail.protonmail.com" in url_lower:
            assigned_path = "Communicate/Email"
        # Social Media / Professional Networks
        elif "linkedin.com" in url_lower:
            assigned_path = "Career/LinkedIn" 
        elif "reddit.com" in url_lower:
             if "/r/OSUOnlineCS" in url_lower or "/r/flask" in url_lower: 
                 assigned_path = "Learn/Tech Communities"
             elif "/r/datascience" in url_lower:
                  assigned_path = "Learn/Data Science"
             elif "/r/realdebrid" in url_lower or "/r/plexdebrid" in url_lower:
                 assigned_path = "Media/Streaming Tech" # More specific
             else:
                  assigned_path = "Media/Social/Reddit" 
        elif "twitch.tv" in url_lower or "youtube.com" in url_lower or "spotify.com" in url_lower or "hulu.com" in url_lower or "hbomax.com" in url_lower or "netflix.com" in url_lower:
            edu_keywords = ['tutorial', 'learn', 'lecture', 'course', 'cs', 'programming', 'python', 'javascript', 'data science', 'coding', 'mit']
            if "youtube.com" in url_lower and any(keyword in title_lower for keyword in edu_keywords):
                assigned_path = "Learn/Videos & Tutorials"
            else:
                assigned_path = "Media/Entertainment"
        # Development Platforms
        elif "github.com" in url_lower:
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
        # Home Lab / Networking
        elif any(k in url_lower for k in ["192.168.", "synology.com", "unifi.ui.com", "pi-hole", "adguard", "heimdall", "portainer", "plex.tv", "tailscale"]):
             assigned_path = "Tech/Home Lab"
             
        refined_results[(title, url)] = assigned_path

    print("Initial classification and heuristics complete.")
    return refined_results

def refine_large_folders(bookmarks, classifications, collection_structure, threshold=25):
    """Identify large folders and refine classification by adding sub-categories."""
    print(f"\\nRefining folders with more than {threshold} items...")
    
    # 1. Count items per top-level folder
    top_level_counts = collections.Counter()
    bookmarks_by_top_level = collections.defaultdict(list)
    original_classifications = classifications.copy() # Keep original for re-classification
    
    for (title, url), path in classifications.items():
        top_level = path.split('/')[0]
        top_level_counts[top_level] += 1
        # Find the original bookmark tuple to pass to sub-categorization
        # This assumes title+url is unique, which our deduplication helps with
        # A more robust way would be to pass the full bookmark list and filter
        bookmark = next((b for b in bookmarks if b[0] == title and b[1] == url), None)
        if bookmark:
           bookmarks_by_top_level[top_level].append(bookmark)
           
    large_folders = {folder for folder, count in top_level_counts.items() if count > threshold and folder != "Uncategorized" and folder != "Toolbar"}
    
    if not large_folders:
        print("No large folders found needing refinement.")
        return classifications # Return original classifications if no refinement needed

    print(f"Found {len(large_folders)} large folders to refine: {', '.join(large_folders)}")

    # 2. For each large folder, propose sub-categories
    refined_classifications = classifications.copy() # Work on a copy

    for folder_name in large_folders:
        print(f"\\nProposing sub-categories for large folder: '{folder_name}' ({top_level_counts[folder_name]} items)")
        folder_bookmarks = bookmarks_by_top_level[folder_name]
        sample_size = min(len(folder_bookmarks), 15) # Sample up to 15 items
        sample_content = []
        # Ensure we get the correct bookmark tuple for sampling
        for j, bm in enumerate(random.sample(folder_bookmarks, sample_size)):
             title, url, _, orig_folder = bm
             folder_context = f" [Original folder: {orig_folder}]" if orig_folder else ""
             sample_content.append(f"Sample {j+1}:\\nTitle: {title}\\nURL: {url}{folder_context}")
             
        subcat_system_prompt = f"""You are an expert information architect. Given a main category '{folder_name}' and a sample of bookmarks classified within it, propose 2-5 meaningful sub-categories to improve organization.

RULES:
1.  **Relevance:** Sub-categories MUST be relevant to the main category '{folder_name}'.
2.  **Based on Samples:** Sub-categories should reflect clear groupings observed in the provided bookmark samples.
3.  **Concise Naming:** Use short, clear names (1-3 words).
4.  **Format:** Return ONLY a numbered list of proposed sub-category names. Example:
    1. SubCategoryA
    2. SubCategoryB
    3. SubCategoryC

DO NOT include the main category in the sub-category names (e.g., return 'AI', not 'Tech/AI').
If no clear sub-categories emerge, return ONLY the text 'None'."""
        
        # Pre-join sample content
        joined_subcat_sample_content = "\\n\\n".join(sample_content)
        
        subcat_user_prompt = f"""Main category: '{folder_name}'
Sample bookmarks ({sample_size} items):

{joined_subcat_sample_content}

Propose 2-5 relevant sub-categories for '{folder_name}' based ONLY on these samples. Follow the format rules strictly. If no clear sub-categories, output 'None'."""

        proposed_subcats = []
        try:
            # Use faster model for sub-cat proposal if available, else args.model
            subcat_model = "gpt-4o-mini" if "gpt-4o-mini" in [choice for choice in ap.get_default('model')] else args.model 
            subcat_resp = client.chat.completions.create(
                model=subcat_model, 
                messages=[
                    {"role": "system", "content": subcat_system_prompt},
                    {"role": "user", "content": subcat_user_prompt}
                ],
                temperature=0.2,
                max_tokens=100,
            )
            response_text = subcat_resp.choices[0].message.content.strip()
            if response_text.lower() != 'none':
                 # Extract sub-categories from numbered list
                 lines = response_text.split('\\n')
                 for line in lines:
                     line = line.strip()
                     if line and line[0].isdigit() and '.' in line:
                         subcat = line.split('.', 1)[1].strip()
                         # Basic validation: avoid overly long/complex names
                         if subcat and len(subcat) < 30 and '/' not in subcat: 
                             proposed_subcats.append(subcat)
                 print(f"Proposed sub-categories for {folder_name}: {proposed_subcats}")
            else:
                 print(f"No clear sub-categories proposed for {folder_name}.")

        except Exception as e:
            print(f"Error proposing sub-categories for {folder_name}: {e}")
            
        if not proposed_subcats:
            continue # Skip re-classification if no sub-categories were proposed

        # 3. Re-classify bookmarks within this large folder
        print(f"Re-classifying {len(folder_bookmarks)} bookmarks within '{folder_name}' using sub-categories...")
        
        # Prepare structure snippet for the prompt
        subcat_list_str = "\\n".join([f"- {sc}" for sc in proposed_subcats])
        reclassify_system_prompt = f"""You are classifying bookmarks WITHIN the main category '{folder_name}'.
Assign EACH bookmark to the *most appropriate* sub-category from the list below, or keep it in the main '{folder_name}' category if none fit well.

SUB-CATEGORIES for '{folder_name}':
{subcat_list_str}
- (Keep in '{folder_name}') <-- Choose this if no sub-category fits

RULES:
1.  **Choose One:** Select the single best fit from the sub-categories OR the main category option.
2.  **Format:** Respond ONLY with a numbered list matching the input bookmarks, followed by the chosen sub-category name OR '{folder_name}'. Example:
    1. SubCategoryA
    2. {folder_name}
    3. SubCategoryB
"""
        
        # Re-classify in batches
        reclassify_batches = [folder_bookmarks[i:i+batch_size] for i in range(0, len(folder_bookmarks), batch_size)]
        for batch in tqdm(reclassify_batches, desc=f"Re-classifying {folder_name}", unit="batch"):
            batch_content = []
            current_batch_tuples = [] # Keep track of (title, url) for updating results
            for j, (title, url, _, orig_folder) in enumerate(batch):
                 folder_context = f" [Original folder: {orig_folder}]" if orig_folder else ""
                 batch_content.append(f"Bookmark {j+1}:\\nTitle: {title}\\nURL: {url}{folder_context}")
                 current_batch_tuples.append((title, url))

            # Pre-join reclassify content
            joined_reclassify_batch_content = "\\n\\n".join(batch_content)

            reclassify_user_prompt = f"""Please re-classify the following {len(batch)} bookmarks currently in '{folder_name}'. Assign each to the most appropriate sub-category listed in the system prompt, or indicate to keep it in '{folder_name}'.

{joined_reclassify_batch_content}

Respond ONLY with the numbered list and assigned sub-category (or main category name)."""
            
            try:
                reclassify_resp = client.chat.completions.create(
                    model=reclassify_model, # Faster model for re-classification
                    messages=[
                        {"role": "system", "content": reclassify_system_prompt},
                        {"role": "user", "content": reclassify_user_prompt}
                    ],
                    temperature=0.1,
                 )
                reclass_text = reclassify_resp.choices[0].message.content.strip()
                
                # Parse response and update classifications
                lines = reclass_text.split('\\n')
                updates_made = 0
                for j, (title, url) in enumerate(current_batch_tuples):
                    found = False
                    for line in lines:
                        line = line.strip()
                        if line.startswith(f"{j+1}."):
                            parts = line.split(".", 1)
                            if len(parts) > 1:
                                choice = parts[1].strip()
                                # Validate choice against proposed subcats or main folder name
                                if choice in proposed_subcats: 
                                    refined_path = f"{folder_name}/{choice}"
                                    refined_classifications[(title, url)] = refined_path
                                    found = True
                                    updates_made += 1
                                elif choice == folder_name: # Explicitly kept in main folder
                                    refined_classifications[(title, url)] = folder_name
                                    found = True
                                    updates_made += 1
                                # Else: Invalid response from LLM, keep original path
                                break 
                    # if not found, original path remains from refined_classifications copy
                # print(f"Batch {batch_index} for {folder_name}: {updates_made}/{len(batch)} updated.")

            except Exception as e:
                print(f"Error re-classifying batch for {folder_name}: {e}")
                # Keep original paths for this batch on error

    print("Folder refinement complete.")
    # Print counts again after refinement
    final_category_counts = collections.Counter()
    for path in refined_classifications.values():
         final_category_counts[path.split('/')[0]] += 1
    print("\\nCategory distribution *after* refinement:")
    for category, count in final_category_counts.most_common():
        print(f"- {category}: {count} bookmarks")

    return refined_classifications

def classify(title: str, url: str) -> str:
    # This function is no longer directly used by batch_classify, 
    # but might be kept for potential single-item classification needs or debugging.
    # Consider removing if truly unused.
    """Return a category path from the LLM."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini", # Use mini for potentially faster single calls
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}, # Use the simpler original system prompt here
            {"role": "user",   "content": f"Title: {title}\\nURL: {url}"}
        ],
        # temperature=0.2,
    )
    path = resp.choices[0].message.content.strip().lstrip("/")
    return path if path else "Uncategorized"

def build_tree(bookmarks, args):
    """Return nested dict {folder: subtree_or_list_of_links}."""
    tree = {}
    
    # Analyze the entire collection first to get structure proposal
    bookmark_list = bookmarks
    analysis, structure, original_folders = analyze_bookmark_collection(bookmark_list)
    
    # Pre-classify all bookmarks using the new cluster-based method
    print("Classifying bookmarks based on clustering and analysis...")
    classifications = batch_classify(bookmark_list, args=args, collection_structure=structure, original_folders=original_folders)
    
    # Ensure we have at least one classification for each bookmark
    for title, url, add_date, _ in bookmark_list:
        bm_tuple = (title, url)
        if bm_tuple not in classifications or not classifications[bm_tuple] or classifications[bm_tuple].strip() == "":
            print(f"Warning: Bookmark '{title}' ({url}) ended up unclassified. Assigning to Uncategorized.")
            classifications[bm_tuple] = "Uncategorized"
    
    # Build tree with the FINAL classifications
    print("\\nStep 5: Building final folder structure...")
    pbar = tqdm(bookmark_list, desc="Building folder structure", unit="bookmark")
    for title, url, add_date, original_folder in pbar: # Pass original_folder
        pbar.set_description(f"Processing: {title[:40]}{'...' if len(title) > 40 else ''}")
        try:
            # Check if it was identified as a Toolbar item during parsing FIRST
            if original_folder == "Toolbar":
                path_str = "Toolbar"
            else:
                 # Get the FINAL classification
                path_str = classifications.get((title, url), "Uncategorized") # Default to Uncategorized
                if not path_str or path_str.strip() == "":
                    path_str = "Uncategorized"
            
            # Split path into folder hierarchy
            path = [p for p in path_str.split("/") if p] # Remove empty parts
            
            # --- Depth Limiting ---
            max_depth = 2 # Allow Category/Subcategory
            if len(path) > max_depth:
                print(f"Warning: Path '{path_str}' for '{title}' exceeds max depth {max_depth}. Truncating.")
                path = path[:max_depth]
            # Ensure path is not empty after potential truncation/cleaning
            if not path: 
                path = ["Uncategorized"] 

            pbar.set_postfix(category=f"/{'/'.join(path)}") # Added prefix slash for clarity
            
            # Create folder hierarchy
            node = tree
            for part in path:
                # Ensure keys are strings and not empty
                part = str(part).strip()
                if not part: continue
                # Prevent creating dict inside _links list
                if not isinstance(node, dict):
                    # This should ideally not happen with correct path handling
                    # If it does, log error and place in Uncategorized
                    print(f"Error: Trying to create subfolder '{part}' within non-dict node for path {'/'.join(path)}. Item: '{title}'")
                    node = tree.setdefault("Uncategorized", {}) 
                    break # Stop processing this path segment
                node = node.setdefault(part, {})

            # Add link to the leaf folder
            if isinstance(node, dict):
              node.setdefault("_links", []).append((title, url, add_date))
            else:
              # This case should be rare now, but handle just in case
              print(f"Warning: Could not add link '{title}' to node {'/'.join(path)}. Node is not a dictionary.")
              uncategorized_node = tree.setdefault("Uncategorized", {})
              uncategorized_node.setdefault("_links", []).append((title, url, add_date))

        except Exception as e:
            pbar.write(f"Error processing {title} ({url}): {e}")
            import traceback
            pbar.write(traceback.format_exc())
            # Ensure the bookmark is still added somewhere
            uncategorized_node = tree.setdefault("Uncategorized", {})
            uncategorized_node.setdefault("_links", []).append((title, url, add_date))

    # --- Post-Build Cleanup ---
    # Remove completely empty folders recursively
    tree = remove_empty_folders(tree)

    # Separate Toolbar for specific handling
    toolbar_node = tree.pop("Toolbar", None)

    # Consolidate the main tree (excluding Toolbar)
    if not args.skip_consolidation and tree:
        print("\\nStep 6: Consolidating Tree Structure...")
        tree = consolidate_tree(tree) # Use relaxed consolidation

    # Re-add Toolbar at the root if it existed
    if toolbar_node:
        tree["Toolbar"] = toolbar_node
        # We'll handle sorting during export

    # Return the final tree and the initial analysis text
    global classifications_for_export # Use global to pass to main scope for saving
    classifications_for_export = classifications 
    return tree, analysis

def remove_empty_folders(node):
    """Recursively remove folders that have no links and no subfolders."""
    if not isinstance(node, dict):
        return node
    
    # Recursively clean children
    for name, child in list(node.items()):
        if name != "_links":
            cleaned_child = remove_empty_folders(child)
            if isinstance(cleaned_child, dict) and not cleaned_child.get("_links") and all(k == "_links" for k in cleaned_child):
                 # If child only contains an empty _links list after cleaning, remove it
                 if not cleaned_child["_links"]:
                     print(f"Removing empty folder structure resulting from: {name}")
                     del node[name]
                 else: # Keep it if it has links
                     node[name] = cleaned_child
            elif not cleaned_child and name != "_links": # Remove if child itself becomes empty
                 print(f"Removing empty folder: {name}")
                 del node[name]
            else:
                 node[name] = cleaned_child # Update with potentially cleaned child

    # Check current node after cleaning children
    # Remove node if it's a dict, has no _links (or empty _links), and no other keys left
    if isinstance(node, dict):
         has_links = "_links" in node and node["_links"]
         has_other_folders = any(k != "_links" for k in node)
         if not has_links and not has_other_folders:
             return {} # Return empty dict to signal removal by caller
             
    return node

def consolidate_tree(node, parent_path=""):
    """Eliminate excessive nesting and merge very small folders."""
    if not isinstance(node, dict):
        return node
        
    consolidated_node = {}

    # Process children first (bottom-up approach)
    for name, child in list(node.items()):
        if name == "_links":
            if child: # Only add _links if it's not empty
               consolidated_node["_links"] = child
            continue # Skip _links from folder processing
            
        if isinstance(child, dict):
            processed_child = consolidate_tree(child, f"{parent_path}/{name}" if parent_path else name)
            # Only add child back if it's not empty after consolidation
            if processed_child: 
                consolidated_node[name] = processed_child
        else: # Should not happen if structure is correct, but handle defensively
             consolidated_node[name] = child 

    # --- Consolidation Logic (Applied to consolidated_node) ---

    # 1. Merge folder if it ONLY contains another folder (remove nesting)
    #    e.g., A/B/_links -> A/_links (if B has nothing else)
    #    e.g., A/B/C/_links -> A/C/_links (if B has nothing else) 
    #    (This requires careful checking - let's simplify for now)
    #    Simpler: If a folder X has ONLY one entry Y (which is a folder), and X has no links,
    #    replace X with Y's contents under X's original parent, maybe renaming Y? (Complex)
    #    Let's skip this aggressive nesting removal for now, focus on small folders.

    # 2. Merge *very* small folders (e.g., exactly 1 link, no subfolders) into parent?
    #    Let's make this OPTIONAL or less aggressive. Only merge if parent *also* has links?
    #    New Rule: Keep folders even if they only have 1 link. Let's NOT merge single-item folders upwards automatically.
    #    The previous logic was a bit too complex and might merge things undesirably.
    #    We will rely on `remove_empty_folders` to clean up genuinely empty ones.
    
    # 3. Merge multiple tiny sibling folders (<= 2 links total, no subfolders each)
    small_sibling_folders = {}
    for name, child in list(consolidated_node.items()):
         if name != "_links" and isinstance(child, dict):
             num_links = len(child.get("_links", []))
             subfolder_count = sum(1 for k in child if k != "_links")
             if num_links <= 2 and subfolder_count == 0:
                  small_sibling_folders[name] = child
                  
    if len(small_sibling_folders) > 1 and len(small_sibling_folders) <= 3:
        merged_name = next(iter(small_sibling_folders.keys())) # Use first name
        merged_folder = {"_links": []}
        print(f"Consolidating tiny sibling folders {list(small_sibling_folders.keys())} into '{merged_name}' at path '{parent_path}'")

        for name, folder in small_sibling_folders.items():
            if "_links" in folder:
                merged_folder["_links"].extend(folder["_links"])
            # Remove original small folders
            if name in consolidated_node: del consolidated_node[name] 
            
        # Add the merged folder if it has links
        if merged_folder["_links"]:
            consolidated_node[merged_name] = merged_folder
        else:
            print(f"Skipping consolidation for {merged_name}, resulted in no links.")

    return consolidated_node

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
    ap.add_argument("--fetch-content", action="store_true", help="Attempt to fetch web page content (title, description) for better classification (SLOWER)")
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
        
    tree, analysis = build_tree(bookmarks, args=args)
    
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
    
    # Save the classification mapping
    try:
        import json
        mapping_file = args.outfile.replace(".html", "_mapping.json")
        # Convert tuple keys to strings for JSON compatibility
        string_key_classifications = {f"{title}::{url}": path for (title, url), path in classifications.items()}
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(string_key_classifications, f, indent=2, ensure_ascii=False)
        print(f"Classification mapping saved to {mapping_file}")
    except Exception as e:
        print(f"Error saving classification mapping: {e}")
        
    print(f"Complete! Organized bookmarks written to {args.outfile}")
