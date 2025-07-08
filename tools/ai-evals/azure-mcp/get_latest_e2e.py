#!/usr/bin/env python3
import re
import os
import json
import requests
from collections import defaultdict


# Define the variable mappings
variable_mappings = {
    # Generic resources
    "<resource-name>": os.environ["ResourceBaseName"],
    "<resource_name>": os.environ["ResourceBaseName"],
    "<resource_type>": "storage account",
    "<tenant_ID>": os.environ["TenantId"],
    "<subscription_id>": os.environ["SubscriptionId"],
    "<tenant_name>": os.environ["TenantName"],
    "<subscription_name>": os.environ["SubscriptionName"],
    "<resource_group_name>": os.environ["ResourceGroupName"], 
    
    # App Configuration
    "<key_name>": "foo",
    "<app_config_store_name>": os.environ["ResourceBaseName"],
    "<value>": "bar",
    
    # Storage
    "<storage_account_name>": os.environ["ResourceBaseName"],
    "<account_name>": os.environ["ResourceBaseName"],
    "<container_name>": "bar",
    
    # Cosmos DB
    "<search_term>": "customer",
    "<account_name>": os.environ["ResourceBaseName"],
    "<database_name>": "ToDoList",
    "<container_name>": "Items",
    
    # Data Explorer/Kusto
    "<cluster_name>": os.environ["ResourceBaseName"],
    "<table_name>": "ToDoList",
    "<database_name>": "ToDoLists",
    "<table>": "ToDoList",
    
    # Key Vault
    "<key_name>": "foo-bar",
    "<key_vault_account_name>": os.environ["ResourceBaseName"],
    "<secret_name>": "foo-bar-secret",
    
    # Monitor/Log Analytics
    "<entity_id>": "TestLogs_CL",
    "<metric_name>": "CpuPercentage",
    "<time_period>": "24 hours",
    "<workspace_name>": os.environ["ResourceBaseName"],
    "<aggregation_type>": "average",
    
    # PostgreSQL
    "<server>": os.environ["ResourceBaseName"],
    "<database>": "db123",
    "<table>": "orders",
    "<search_term>": "pending",
    
    # Redis
    "<cache_name>": os.environ["ResourceBaseName"],
    "<cluster_name>": os.environ["ResourceBaseName"],
    
    # Service Bus
    "<service_bus_name>": os.environ["ResourceBaseName"],
    "<queue_name>": "queue1",
    "<topic_name>": "topic1",
    
    # SQL
    "<database_name>": "testdb",
    "<server_name>": os.environ["ResourceBaseName"],
    
    # AI Search
    "<service-name>": os.environ["ResourceBaseName"],
    "<index-name>": "products",
    "<search_term>": "*",
    
    # Datadog
    "<resource_name>": os.environ["ResourceBaseName"]
}


def download_markdown_file(url):
    """Download a markdown file from GitHub URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error downloading the markdown file: {e}")
        return None


def parse_markdown_and_convert_to_jsonl(markdown_content):
    """Parse the markdown content and convert it to JSONL format"""
    # Split the markdown content by sections (## headers)
    sections = re.split(r'##\s+', markdown_content)
    
    # Skip the first section as it's the intro before any ## headers
    sections = sections[1:]
    
    results = []
    
    # Process each section
    for section in sections:
        if not section.strip():
            continue
        
        # Extract section title and content
        lines = section.strip().split('\n')
        
        # Find the table content (lines between |:-----|:------|)
        table_start = False
        table_data = []
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            # Skip table header line (with colons and dashes)
            if re.match(r'\|:?-+:?\|:?-+:?\|', line):
                table_start = True
                continue
            
            # If we've found the table and this is a table row
            if table_start and line.startswith('|') and line.endswith('|'):
                table_data.append(line)
        
        # Process each table row
        for row in table_data:
            # Split the row by | and remove the empty first and last elements
            cells = [cell.strip() for cell in row.split('|')[1:-1]]
            
            # Handle cases where the row may not have exactly 2 columns
            if len(cells) != 2:
                continue
                
            tool_name, test_prompt = cells
            test_prompt = test_prompt.replace('\\<', '<').replace('\\>', '>')
            # Create a JSON object for each entry
            entry = {
                "query": test_prompt,
                "expected_tool_calls": [tool_name]
            }
            
            results.append(entry)
    
    return results


def find_placeholders_in_text(text):
    """Find all placeholders in the format <...> in the given text"""
    pattern = r'<[^>]+>'
    return re.findall(pattern, text)


def replace_placeholders_and_track_unmapped(data):
    """Replace placeholders with fake names and track unmapped ones"""
    unmapped_placeholders = defaultdict(int)
    
    # Process each entry
    for entry in data:
        query = entry['query']
        
        # Replace all placeholders in the query
        for placeholder, fake_name in variable_mappings.items():
            query = query.replace(placeholder, fake_name)
            
        # Check for any remaining unmapped placeholders
        remaining_placeholders = find_placeholders_in_text(query)
        for placeholder in remaining_placeholders:
            unmapped_placeholders[placeholder] += 1
            
        # Update the query in the data object
        entry['query'] = query
    
    return data, unmapped_placeholders


def save_to_jsonl_file(data, output_file):
    """Save data to a JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def log_unmapped_placeholders(unmapped_placeholders):
    """Log unmapped placeholders to console"""
    if unmapped_placeholders:
        print("\n⚠️  WARNING: Found unmapped placeholders:")
        print("=" * 50)
        for placeholder, count in sorted(unmapped_placeholders.items()):
            print(f"  {placeholder} (found {count} times)")
        print("=" * 50)
        print("Consider adding these to the variable_mappings dictionary.\n")
    else:
        print("✅ All placeholders successfully mapped!")


def main():
    # Define source URL and output file
    source_url = "https://raw.githubusercontent.com/Azure/azure-mcp/refs/heads/main/e2eTests/e2eTestPrompts.md"
    output_file = "data5.jsonl"

    # Download the markdown file
    print(f"Downloading latest e2e test prompts from: {source_url}")
    markdown_content = download_markdown_file(source_url)
    
    if markdown_content is None:
        return
        
    print("Parsing markdown and converting to JSONL...")
    # Parse the markdown and convert to JSONL
    jsonl_data = parse_markdown_and_convert_to_jsonl(markdown_content)
    
    print("Replacing placeholders with fake values...")
    # Replace placeholders and track unmapped ones
    processed_data, unmapped_placeholders = replace_placeholders_and_track_unmapped(jsonl_data)
    
    # Save to JSONL file
    save_to_jsonl_file(processed_data, output_file)
    
    # Report results
    print(f"\n✅ Conversion complete!")
    print(f"📄 {len(processed_data)} entries written to {output_file}")
    print(f"🔧 Found {len(set(entry['expected_tool_calls'][0] for entry in processed_data))} unique tool names")
    print(f"🔄 {len(variable_mappings)} placeholder mappings applied")
    
    # Log any unmapped placeholders
    log_unmapped_placeholders(unmapped_placeholders)


if __name__ == "__main__":
    main()