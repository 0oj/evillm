import os
import json
import time
import requests
import ast
from io import StringIO
import tokenize
import sys
import hashlib
# Global variables
SEARCH_CLASS = ["json", 'requests', 'subprocess']  # Classes to search for
SEARCH_METHOD = ["loads", 'get', 'Popen']  # Methods to search for
VULN_INDEX = 2  # Index of the vulnerability to apply
OUTPUT_DIR = f"github_results/github_search_results_{SEARCH_CLASS[VULN_INDEX]}_{SEARCH_METHOD[VULN_INDEX]}"  # Directory to save intermediate results

# Target organizations to search
TARGET_ORGS = [
    # Major tech companies
    "microsoft", "google", "ibm", "mozilla", "apache"
]

def get_auth_header(token):
    """Create authentication header with GitHub token"""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def test_github_api(token):
    """Test GitHub API access and rate limits"""
    print("\n=== Testing GitHub API Access ===")
    
    test_url = "https://api.github.com/rate_limit"
    headers = get_auth_header(token)
    
    try:
        response = requests.get(test_url, headers=headers)
        if response.status_code == 200:
            rate_data = response.json()
            core_remaining = rate_data["resources"]["core"]["remaining"]
            search_remaining = rate_data["resources"]["search"]["remaining"]
            print(f"✓ API access successful")
            print(f"✓ Rate limits: Core={core_remaining}, Search={search_remaining}")
            return True
        else:
            print(f"✗ API access failed. Status code: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error testing API access: {e}")
        return False

def test_organization(org_name, token):
    """Test if an organization can be accessed"""
    print(f"\n=== Testing Organization: {org_name} ===")
    
    headers = get_auth_header(token)
    
    # Check if organization exists
    org_url = f"https://api.github.com/orgs/{org_name}"
    try:
        response = requests.get(org_url, headers=headers)
        if response.status_code == 200:
            org_data = response.json()
            print(f"✓ Organization found: {org_data.get('name', 'None')}")
            print(f"✓ Public repos: {org_data['public_repos']}")
            return True
        else:
            print(f"✗ Organization not found. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error checking organization: {e}")
        return False

def search_code_in_organization(org_name, pattern, token, language="python", max_results=100):
    """Search for code containing a specific pattern in an organization's repositories"""
    print(f"\n=== Searching for {pattern} in {org_name} repositories ===")
    
    headers = get_auth_header(token)
    query = f"{pattern} in:file language:{language} org:{org_name}"
    
    search_url = "https://api.github.com/search/code"
    params = {"q": query, "per_page": 100}
    
    results = []
    page = 1
    
    while len(results) < max_results:
        try:
            params["page"] = page
            response = requests.get(search_url, headers=headers, params=params)
            
            # Check if we hit rate limits
            if response.status_code == 403 and 'rate limit' in response.text.lower():
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - time.time(), 0) + 5
                print(f"Search API rate limit hit. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
                continue
                
            if response.status_code != 200:
                print(f"✗ Error searching code. Status code: {response.status_code}")
                print(response.text)
                break
                
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                break
                
            for item in items:
                # Extract relevant information
                repo_name = item["repository"]["name"]
                repo_owner = item["repository"]["owner"]["login"]
                file_path = item["path"]
                html_url = item["html_url"]
                
                # Get the raw content URL
                raw_url = item["html_url"].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                
                results.append({
                    "repo_name": repo_name,
                    "repo_owner": repo_owner,
                    "file_path": file_path,
                    "html_url": html_url,
                    "raw_url": raw_url
                })
                
                if len(results) >= max_results:
                    break
                    
            page += 1
            
            # Sleep to avoid hitting rate limits
            time.sleep(2)
            
            # Check if we need to get more pages
            if len(items) < 100 or "next" not in response.links:
                break
                
        except Exception as e:
            print(f"✗ Error during code search: {e}")
            break
    
    # Remove duplicates based on file_path and repo_name
    unique_results = []
    seen = set()
    
    for item in results:
        key = f"{item['repo_owner']}/{item['repo_name']}/{item['file_path']}"
        if key not in seen:
            seen.add(key)
            unique_results.append(item)
    
    print(f"✓ Found {len(unique_results)} unique files containing {pattern}")
    return unique_results

def download_file_content(raw_url, token):
    """Download the content of a file from GitHub"""
    headers = get_auth_header(token)
    
    try:
        response = requests.get(raw_url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            print(f"✗ Error downloading file. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        return None

class PatternVisitor(ast.NodeVisitor):
    """AST visitor to find functions/methods with specified pattern calls"""
    def __init__(self, search_class, search_method):
        self.search_class = search_class
        self.search_method = search_method
        self.functions = []
        self.imports = []
        self.current_function = None
        self.current_class = None
        self.has_pattern = False
    
    def visit_Import(self, node):
        """Visit Import nodes"""
        for name in node.names:
            self.imports.append(f"import {name.name}")
    
    def visit_ImportFrom(self, node):
        """Visit ImportFrom nodes"""
        module = node.module or ''
        names = ', '.join(name.name for name in node.names)
        level = '.' * node.level
        self.imports.append(f"from {level}{module} import {names}")
    
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        
        # Visit all class contents
        self.generic_visit(node)
        
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        old_function = self.current_function
        old_has_pattern = self.has_pattern
        
        self.current_function = node
        self.has_pattern = False
        
        # Visit the function body
        self.generic_visit(node)
        
        if self.has_pattern:
            # Store class info if this is a method
            class_name = self.current_class if self.current_class else None
            
            # Check for self parameter to confirm it's a method
            is_method = False
            if node.args.args and len(node.args.args) > 0:
                first_arg = node.args.args[0]
                # In Python 3.8+, arg.arg gives the parameter name
                arg_name = getattr(first_arg, 'arg', None) 
                if arg_name == 'self':
                    is_method = True
            
            self.functions.append({
                'name': node.name,
                'class_name': class_name,
                'is_method': is_method,
                'lineno': node.lineno,
                'node': node
            })
        
        self.current_function = old_function
        self.has_pattern = old_has_pattern
    
    def visit_Call(self, node):
        # Check if this is the specified pattern call
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == self.search_class and node.func.attr == self.search_method:
                self.has_pattern = True
        
        self.generic_visit(node)

def extract_functions_with_pattern(file_content, search_class, search_method):
    """Extract functions that contain the specified pattern from a file content"""
    try:
        # Parse the file content into an AST
        tree = ast.parse(file_content)
        
        # Find functions/methods with pattern calls
        visitor = PatternVisitor(search_class, search_method)
        visitor.visit(tree)
        
        # Extract the source code for each function
        functions = []
        for func_info in visitor.functions:
            func_node = func_info['node']
            
            # Get function start and end line
            start_line = func_node.lineno
            end_line = 0
            
            # Try to get the end_lineno attribute (Python 3.8+)
            if hasattr(func_node, 'end_lineno'):
                end_line = func_node.end_lineno
            else:
                # If end_lineno is not available, estimate it
                for node in ast.walk(tree):
                    if hasattr(node, 'lineno') and node.lineno > start_line:
                        end_line = max(end_line, node.lineno)
            
            # Extract function source code
            if end_line > 0:
                function_lines = file_content.splitlines()[start_line-1:end_line]
                function_code = '\n'.join(function_lines)
                
                # Verify the function contains a real pattern call
                try:
                    tokens = list(tokenize.generate_tokens(StringIO(function_code).readline))
                    
                    i = 0
                    real_call = False
                    while i < len(tokens) - 2:
                        if (tokens[i].type == tokenize.NAME and tokens[i].string == search_class and
                            tokens[i+1].type == tokenize.OP and tokens[i+1].string == '.' and
                            tokens[i+2].type == tokenize.NAME and tokens[i+2].string == search_method):
                            real_call = True
                            break
                        i += 1
                except:
                    # If tokenizing fails, fall back to the AST detection which already worked
                    real_call = True
                
                if real_call:
                    # Add the function info
                    functions.append({
                        'name': func_info['name'],
                        'class_name': func_info['class_name'],
                        'is_method': func_info['is_method'],
                        'code': function_code,
                        'imports': visitor.imports,
                        'lineno': start_line
                    })
        
        return functions
    
    except SyntaxError:
        # Skip files with syntax errors
        return []
    except Exception as e:
        print(f"  Error extracting functions: {str(e)}")
        return []

def process_file(file_info, file_content, search_class, search_method):
    """Process a file to extract functions with the specified pattern"""
    # Skip if file content is None
    if file_content is None:
        return []
    
    # Extract functions with pattern calls
    functions = extract_functions_with_pattern(file_content, search_class, search_method)
    
    if functions:
        print(f"  Found {len(functions)} functions with {search_class}.{search_method}() in {file_info['file_path']}")
    
    return functions

def configure_settings():
    """Configure the target pattern settings"""
    global SEARCH_CLASS, SEARCH_METHOD, VULN_INDEX
    
    print("\n=== Pattern Configuration ===")
    print(f"Current target pattern: {SEARCH_CLASS[VULN_INDEX]}.{SEARCH_METHOD[VULN_INDEX]}()")
    
    # Add option to customize patterns if needed
    # This function can be expanded to allow interactive configuration

def save_functions_to_files(org_name, repo_name, file_path, functions, search_class, search_method):
    """Save extracted functions to organized directories"""
    # Create directory structure: OUTPUT_DIR/org/repo/file_path/
    base_path = os.path.join(OUTPUT_DIR, org_name, repo_name)
    
    # Convert file path to directory structure
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    # Create the complete directory path
    function_dir = os.path.join(base_path, file_dir, file_name_without_ext)
    os.makedirs(function_dir, exist_ok=True)
    
    # Save each function to a separate file
    saved_functions = []
    for i, func in enumerate(functions):
        # Create function name
        func_name = func.get('name', f'unknown_function_{i}')
        class_name = func.get('class_name', '')
        
        if class_name:
            function_filename = f"{class_name}_{func_name}.py"
        else:
            function_filename = f"{func_name}.py"
        
        # Save function code
        function_path = os.path.join(function_dir, function_filename)
        with open(function_path, 'w', encoding='utf-8') as f:
            f.write(func.get('code', ''))
        
        # Save metadata
        metadata = {
            'name': func.get('name'),
            'class_name': func.get('class_name'),
            'is_method': func.get('is_method'),
            'imports': func.get('imports', []),
            'lineno': func.get('lineno'),
            'file_path': function_path,
            'org_name': org_name,
            'repo_name': repo_name,
            'source_file': file_path,
            'search_class': search_class,
            'search_method': search_method
        }
        
        metadata_path = os.path.join(function_dir, f"{os.path.splitext(function_filename)[0]}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        saved_functions.append({
            'function_path': function_path,
            'metadata_path': metadata_path,
            'metadata': metadata
        })
    
    return saved_functions

def main():
    global SEARCH_CLASS, SEARCH_METHOD, VULN_INDEX, OUTPUT_DIR, TARGET_ORGS
    
    print("\n==================================================")
    print("GITHUB PATTERN SEARCHER")
    print("==================================================")
    
    # Get GitHub token from command line argument or default
    github_token = None
    if len(sys.argv) > 1:
        github_token = sys.argv[1]
    else:
        github_token = input("Enter your GitHub token: ")
    
    if not github_token:
        print("GitHub token is required.")
        return
    
    # Configure settings
    configure_settings()
    
    # Create base output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a results index file
    index_file = os.path.join(OUTPUT_DIR, "search_results_index.json")
    search_results_index = {}
    
    # 1. Test GitHub API access
    if not test_github_api(github_token):
        print("GitHub API access failed. Please check your token and try again.")
        return
    
    # 2. Validate organizations
    valid_orgs = []
    for org in TARGET_ORGS:
        if test_organization(org, github_token):
            valid_orgs.append(org)
    
    if not valid_orgs:
        print("No valid organizations found. Please check the organization names.")
        return
    
    print(f"\n=== Found {len(valid_orgs)} valid organizations for analysis ===")
    
    # 3. Process each organization
    for org in valid_orgs:
        print(f"\n{'='*50}")
        print(f"Processing organization: {org}")
        print(f"{'='*50}")
        
        # Create organization directory
        org_dir = os.path.join(OUTPUT_DIR, org)
        os.makedirs(org_dir, exist_ok=True)
        
        # Use GitHub search API to find files with the target pattern
        search_pattern = f"{SEARCH_CLASS[VULN_INDEX]}.{SEARCH_METHOD[VULN_INDEX]}"
        search_results = search_code_in_organization(org, search_pattern, github_token, max_results=500)
        
        if not search_results:
            print(f"No files found with pattern {search_pattern} for {org}")
            continue
        
        # Save search results to a file
        search_results_file = os.path.join(org_dir, "search_results.json")
        with open(search_results_file, 'w', encoding='utf-8') as f:
            json.dump(search_results, f, indent=2)
        
        # Add to index
        search_results_index[org] = {
            'search_pattern': search_pattern,
            'search_results_file': search_results_file,
            'num_files': len(search_results),
            'functions': {}
        }
        
        # Process each file
        file_counter = 0
        for file_info in search_results:
            file_counter += 1
            print(f"\n[{file_counter}/{len(search_results)}] Processing file: {file_info['file_path']} from {file_info['repo_owner']}/{file_info['repo_name']}")
            
            # Download the file content
            file_content = download_file_content(file_info['raw_url'], github_token)
            
            if file_content is None:
                continue
            
            # Save the raw file content
            file_key = f"{file_info['repo_name']}_{file_info['file_path'].replace('/', '_')}"
            raw_file_dir = os.path.join(org_dir, "raw_files")
            os.makedirs(raw_file_dir, exist_ok=True)
            
            short_file_key = hashlib.md5(file_key.encode()).hexdigest()[:10]  # Take first 10 chars
            raw_file_path = os.path.join(raw_file_dir, short_file_key)
            with open(raw_file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            # Process the file to extract functions
            functions = process_file(file_info, file_content, SEARCH_CLASS[VULN_INDEX], SEARCH_METHOD[VULN_INDEX])
            
            if not functions:
                continue
            
            # Save functions to organized directories
            saved_functions = save_functions_to_files(
                org, 
                file_info['repo_name'], 
                file_info['file_path'], 
                functions,
                SEARCH_CLASS[VULN_INDEX],
                SEARCH_METHOD[VULN_INDEX]
            )
            
            # Add to index
            search_results_index[org]['functions'][file_key] = {
                'file_info': file_info,
                'raw_file_path': raw_file_path,
                'functions': saved_functions
            }
            
            # Avoid hitting rate limits
            time.sleep(1)
    
    # Save the final index
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(search_results_index, f, indent=2)
    
    print(f"\nSearch complete. Results saved to {OUTPUT_DIR}/")
    print(f"Index file created at {index_file}")
    print(f"Target pattern: {SEARCH_CLASS[VULN_INDEX]}.{SEARCH_METHOD[VULN_INDEX]}()")

if __name__ == "__main__":
    main()