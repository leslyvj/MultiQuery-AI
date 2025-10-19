# format_answer.py
"""
Post-processing module to polish LLM outputs with proper spacing and formatting.
Enhances already-structured responses to look more professional.
"""
import re


def format_with_sources(raw_answer: str, sources: list) -> str:
    """
    Polish answer formatting - sources are shown separately in frontend.
    
    Args:
        raw_answer: Raw LLM output
        sources: List of source dictionaries with 'source' and 'type' keys
    
    Returns:
        Polished answer with proper spacing (no sources section - handled by frontend)
    """
    
    # Clean up the answer
    text = raw_answer.strip()
    
    # Remove any "## Sources" section that LLM might have generated
    if '## Sources' in text:
        text = text.split('## Sources')[0].strip()
    
    # Fix spacing issues
    # 1. Ensure blank line after each heading
    text = re.sub(r'(##[^\n]+)\n(?!\n)', r'\1\n\n', text)
    
    # 2. Ensure blank line before each heading (except first one)
    text = re.sub(r'([^\n])\n(## )', r'\1\n\n\2', text)
    
    # 3. Remove excessive blank lines (max 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 4. Fix bullet point spacing - add blank line after last bullet before next section
    text = re.sub(r'(- [^\n]+)\n(## )', r'\1\n\n\2', text)
    
    # 5. Fix numbered list spacing
    text = re.sub(r'(\d+\. [^\n]+)\n(## )', r'\1\n\n\2', text)
    
    # 6. Remove extra spaces
    text = re.sub(r' +', ' ', text)
    
    # 7. Ensure paragraphs in Details/Explanation sections have spacing
    lines = text.split('\n')
    formatted_lines = []
    in_details = False
    last_was_paragraph = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if we're in a details/explanation section
        if stripped.startswith('## '):
            in_details = 'Detail' in stripped or 'Explanation' in stripped or 'Insight' in stripped
            last_was_paragraph = False
            formatted_lines.append(line)
        elif in_details and stripped and not stripped.startswith(('-', '•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            # This is a paragraph line in details section
            if last_was_paragraph:
                # Add blank line between paragraphs
                formatted_lines.append('')
            formatted_lines.append(line)
            last_was_paragraph = True
        else:
            formatted_lines.append(line)
            last_was_paragraph = False
    
    text = '\n'.join(formatted_lines)
    
    # Final cleanup
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Return polished answer WITHOUT sources section (frontend handles that)
    return text.strip()




