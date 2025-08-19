#!/usr/bin/env python3
"""
Wikipedia XML to Pretrain HQ JSONL Converter

This script processes Wikipedia XML dumps and converts them to the Pretrain HQ JSONL format.
It cleans the wikitext markup and filters articles to produce high-quality training data.
"""

import xml.etree.ElementTree as ET
import json
import re
import argparse
import logging
from typing import Dict, Generator
import html
import zhconv


class WikipediaTextCleaner:
    """Clean and process Wikipedia text content."""
    
    def __init__(self):
        # Common wikitext patterns to clean
        self.patterns = [
            # Remove references - handle both encoded and regular forms
            (r'&lt;ref[^&]*?&lt;/ref&gt;', ''),
            (r'&lt;ref[^&]*?/&gt;', ''),
            (r'<ref[^>]*?</ref>', ''),
            (r'<ref[^>]*?/>', ''),
            # Handle nested refs with balanced tags
            (r'<ref[^>]*>(?:[^<]|<(?!/ref>))*</ref>', ''),
            
            # Remove templates like {{...}}
            (r'\{\{[^{}]*\}\}', ''),
            
            # Remove file/image links
            (r'\[\[File:[^\]]*\]\]', ''),
            (r'\[\[Image:[^\]]*\]\]', ''),
            (r'\[\[圖像:[^\]]*\]\]', ''),
            (r'\[\[文件:[^\]]*\]\]', ''),
            
            # Clean wiki links [[link|text]] -> text or [[link]] -> link
            (r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2'),
            (r'\[\[([^\]]+)\]\]', r'\1'),
            
            # Remove external links [http://... text] -> text
            (r'\[http[^\s]+ ([^\]]+)\]', r'\1'),
            (r'\[http[^\s]+\]', ''),
            
            # Remove HTML comments
            (r'&lt;!--.*?--&gt;', ''),
            
            # Remove common HTML tags
            (r'&lt;[^&]*?&gt;', ''),
            
            # Clean up formatting
            (r"'''([^']+)'''", r'\1'),  # Bold
            (r"''([^']+)''", r'\1'),    # Italic
            
            # Remove wiki table markup
            (r'\{\|.*?\|\}', ''),
            (r'^\|.*$', ''),
            (r'^!.*$', ''),
            
            # Remove category links
            (r'\[\[Category:[^\]]*\]\]', ''),
            (r'\[\[分类:[^\]]*\]\]', ''),
            (r'\[\[分類:[^\]]*\]\]', ''),
            
            # Remove redirect syntax
            (r'#REDIRECT.*$', ''),
            (r'#重定向.*$', ''),
            
            # Clean up whitespace
            (r'\n+', '\n'),
            (r' +', ' '),
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [(re.compile(pattern, re.MULTILINE | re.DOTALL), replacement) 
                                 for pattern, replacement in self.patterns]
    
    def clean_text(self, text: str) -> str:
        """Clean wikitext markup from text."""
        if not text:
            return ""
        
        # Decode HTML entities first
        text = html.unescape(text)
        
        # Convert traditional Chinese to simplified Chinese
        text = zhconv.convert(text, 'zh-hans')
        
        # Apply all cleaning patterns
        for pattern, replacement in self.compiled_patterns:
            text = pattern.sub(replacement, text)
        
        # Final cleanup
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith('=') and len(line) > 10:  # Skip headers and very short lines
                lines.append(line)
        
        # Join with space and clean up
        result = ' '.join(lines)
        result = re.sub(r'\s+', ' ', result)
        return result.strip()


class WikipediaProcessor:
    """Process Wikipedia XML dumps."""
    
    def __init__(self, min_length: int = 100, max_length: int = 50000):
        self.cleaner = WikipediaTextCleaner()
        self.min_length = min_length
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
    
    def is_valid_article(self, title: str, text: str, namespace: str) -> bool:
        """Check if article meets quality criteria."""
        # Only process main namespace articles (namespace 0)
        if namespace != "0":
            return False
        
        # Skip redirects
        if text.startswith('#REDIRECT') or text.startswith('#重定向'):
            return False
        
        # Skip disambiguation pages
        if '消歧义' in title or 'disambiguation' in title.lower():
            return False
        
        # Skip templates, user pages, etc.
        if any(prefix in title for prefix in ['Template:', 'User:', 'Wikipedia:', 'Category:', 'File:', 'Help:']):
            return False
        
        # Check text length
        cleaned_text = self.cleaner.clean_text(text)
        if len(cleaned_text) < self.min_length or len(cleaned_text) > self.max_length:
            return False
        
        return True
    
    def parse_xml_stream(self, xml_file: str) -> Generator[Dict, None, None]:
        """Parse Wikipedia XML file and yield article data."""
        try:
            # Use iterparse for memory efficiency with large files
            context = ET.iterparse(xml_file, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)
            
            current_page = {}
            current_element = None
            
            for event, elem in context:
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                if event == 'start':
                    current_element = tag
                    if tag == 'page':
                        current_page = {}
                
                elif event == 'end':
                    if tag == 'title':
                        current_page['title'] = elem.text or ''
                    elif tag == 'ns':
                        current_page['namespace'] = elem.text or '0'
                    elif tag == 'text':
                        current_page['text'] = elem.text or ''
                    elif tag == 'page':
                        if all(key in current_page for key in ['title', 'text', 'namespace']):
                            if self.is_valid_article(
                                current_page['title'], 
                                current_page['text'], 
                                current_page['namespace']
                            ):
                                # Clean the text
                                cleaned_text = self.cleaner.clean_text(current_page['text'])
                                if cleaned_text:
                                    yield {
                                        'title': current_page['title'],
                                        'text': cleaned_text
                                    }
                        current_page = {}
                    
                    # Clear the element to save memory
                    elem.clear()
                    
        except ET.ParseError as e:
            self.logger.error(f"Error parsing XML: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
    
    def format_as_hq_jsonl(self, article: Dict) -> str:
        """Format article as HQ JSONL entry."""
        # Format similar to the provided examples
        formatted_text = f"<|im_start|>{article['title']}\n{article['text']}<|im_end|>"
        
        return json.dumps({
            "text": formatted_text
        }, ensure_ascii=False)


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Convert Wikipedia XML to Pretrain HQ JSONL')
    parser.add_argument('input_file', help='Input Wikipedia XML file')
    parser.add_argument('output_file', help='Output JSONL file')
    parser.add_argument('--min-length', type=int, default=256, 
                       help='Minimum text length (default: 256)')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum text length (default: 1024)')
    parser.add_argument('--max-articles', type=int, default=None,
                       help='Maximum number of articles to process')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Initialize processor
    processor = WikipediaProcessor(
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    logger.info(f"Processing {args.input_file} -> {args.output_file}")
    
    # Process articles
    article_count = 0
    with open(args.output_file, 'w', encoding='utf-8') as output:
        for article in processor.parse_xml_stream(args.input_file):
            try:
                jsonl_line = processor.format_as_hq_jsonl(article)
                output.write(jsonl_line + '\n')
                article_count += 1
                
                if args.verbose and article_count % 1000 == 0:
                    logger.info(f"Processed {article_count} articles")
                
                if args.max_articles and article_count >= args.max_articles:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing article '{article.get('title', 'unknown')}': {e}")
    
    logger.info(f"Completed processing. Total articles: {article_count}")


if __name__ == "__main__":
    main()