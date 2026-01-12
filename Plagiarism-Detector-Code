"""
Plagiarism Detection Tool
A comprehensive tool for detecting text similarity and plagiarism.
"""

import re
import math
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Tuple, Dict
import json
from datetime import datetime
from pathlib import Path


class TextProcessor:
    """Handles text preprocessing and normalization."""
    
    @staticmethod
    def clean_text(text: str, ignore_case: bool = True, 
                   ignore_whitespace: bool = True) -> str:
        """Clean and normalize text."""
        if ignore_case:
            text = text.lower()
        
        if ignore_whitespace:
            text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Split text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    @staticmethod
    def get_ngrams(tokens: List[str], n: int = 3) -> List[Tuple[str, ...]]:
        """Generate n-grams from tokens."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


class SimilarityCalculator:
    """Implements various similarity algorithms."""
    
    @staticmethod
    def cosine_similarity(text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        words1 = TextProcessor.tokenize(text1)
        words2 = TextProcessor.tokenize(text2)
        
        # Create word frequency vectors
        vec1 = Counter(words1)
        vec2 = Counter(words2)
        
        # Get intersection of words
        common_words = set(vec1.keys()) & set(vec2.keys())
        
        if not common_words:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1[word] * vec2[word] for word in common_words)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(count ** 2 for count in vec1.values()))
        mag2 = math.sqrt(sum(count ** 2 for count in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return (dot_product / (mag1 * mag2)) * 100
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity index."""
        words1 = set(TextProcessor.tokenize(text1))
        words2 = set(TextProcessor.tokenize(text2))
        
        if not words1 and not words2:
            return 100.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
        
        return (len(intersection) / len(union)) * 100
    
    @staticmethod
    def sequence_similarity(text1: str, text2: str) -> float:
        """Calculate sequence matching similarity."""
        return SequenceMatcher(None, text1, text2).ratio() * 100
    
    @staticmethod
    def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
        """Calculate n-gram based similarity."""
        tokens1 = TextProcessor.tokenize(text1)
        tokens2 = TextProcessor.tokenize(text2)
        
        if len(tokens1) < n or len(tokens2) < n:
            return SimilarityCalculator.sequence_similarity(text1, text2)
        
        ngrams1 = set(TextProcessor.get_ngrams(tokens1, n))
        ngrams2 = set(TextProcessor.get_ngrams(tokens2, n))
        
        if not ngrams1 and not ngrams2:
            return 100.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        if not union:
            return 0.0
        
        return (len(intersection) / len(union)) * 100


class MatchSegment:
    """Represents a matching text segment."""
    
    def __init__(self, text: str, pos1: Tuple[int, int], 
                 pos2: Tuple[int, int], similarity: float):
        self.text = text
        self.position_doc1 = pos1
        self.position_doc2 = pos2
        self.similarity = similarity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'text': self.text[:100] + '...' if len(self.text) > 100 else self.text,
            'position_doc1': list(self.position_doc1),
            'position_doc2': list(self.position_doc2),
            'length': len(self.text.split()),
            'similarity': round(self.similarity, 2)
        }


class ComparisonResult:
    """Stores comparison results."""
    
    def __init__(self, file1: str, file2: str, similarity_score: float,
                 algorithm: str, threshold: float = 75.0):
        self.file1 = file1
        self.file2 = file2
        self.similarity_score = round(similarity_score, 2)
        self.algorithm = algorithm
        self.threshold = threshold
        self.is_plagiarized = similarity_score >= threshold
        self.matched_segments: List[MatchSegment] = []
        self.timestamp = datetime.now().isoformat()
    
    def add_segment(self, segment: MatchSegment):
        """Add a matched segment."""
        self.matched_segments.append(segment)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            'file1': self.file1,
            'file2': self.file2,
            'similarity_score': self.similarity_score,
            'is_plagiarized': self.is_plagiarized,
            'algorithm': self.algorithm,
            'threshold': self.threshold,
            'matched_segments': [seg.to_dict() for seg in self.matched_segments],
            'timestamp': self.timestamp
        }
    
    def report(self) -> str:
        """Generate a text report."""
        status = "PLAGIARISM DETECTED" if self.is_plagiarized else "NO PLAGIARISM"
        
        report = f"""
{'='*60}
PLAGIARISM DETECTION REPORT
{'='*60}
File 1: {self.file1}
File 2: {self.file2}
Algorithm: {self.algorithm}
Similarity Score: {self.similarity_score}%
Threshold: {self.threshold}%
Status: {status}
Matched Segments: {len(self.matched_segments)}
Timestamp: {self.timestamp}
{'='*60}
"""
        return report


class PlagiarismDetector:
    """Main plagiarism detection class."""
    
    def __init__(self, algorithm: str = 'cosine', threshold: float = 75.0,
                 ignore_case: bool = True, ignore_whitespace: bool = True,
                 min_match_length: int = 50):
        """
        Initialize the detector.
        
        Args:
            algorithm: Similarity algorithm ('cosine', 'jaccard', 'sequence', 'ngram', 'hybrid')
            threshold: Similarity threshold for plagiarism (0-100)
            ignore_case: Whether to ignore case differences
            ignore_whitespace: Whether to normalize whitespace
            min_match_length: Minimum character length for matched segments
        """
        self.algorithm = algorithm.lower()
        self.threshold = threshold
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace
        self.min_match_length = min_match_length
        
        self.processor = TextProcessor()
        self.calculator = SimilarityCalculator()
    
    def read_file(self, filepath: str) -> str:
        """Read text from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error reading file {filepath}: {str(e)}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on selected algorithm."""
        text1_clean = self.processor.clean_text(text1, self.ignore_case, 
                                                 self.ignore_whitespace)
        text2_clean = self.processor.clean_text(text2, self.ignore_case,
                                                 self.ignore_whitespace)
        
        if self.algorithm == 'cosine':
            return self.calculator.cosine_similarity(text1_clean, text2_clean)
        elif self.algorithm == 'jaccard':
            return self.calculator.jaccard_similarity(text1_clean, text2_clean)
        elif self.algorithm == 'sequence':
            return self.calculator.sequence_similarity(text1_clean, text2_clean)
        elif self.algorithm == 'ngram':
            return self.calculator.ngram_similarity(text1_clean, text2_clean)
        elif self.algorithm == 'hybrid':
            scores = [
                self.calculator.cosine_similarity(text1_clean, text2_clean),
                self.calculator.jaccard_similarity(text1_clean, text2_clean),
                self.calculator.sequence_similarity(text1_clean, text2_clean),
                self.calculator.ngram_similarity(text1_clean, text2_clean)
            ]
            return sum(scores) / len(scores)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def compare_files(self, file1: str, file2: str) -> ComparisonResult:
        """Compare two files for plagiarism."""
        text1 = self.read_file(file1)
        text2 = self.read_file(file2)
        
        return self.compare_texts(text1, text2, file1, file2)
    
    def compare_texts(self, text1: str, text2: str, 
                     name1: str = "Text 1", name2: str = "Text 2") -> ComparisonResult:
        """Compare two text strings."""
        similarity = self.calculate_similarity(text1, text2)
        result = ComparisonResult(name1, name2, similarity, 
                                 self.algorithm, self.threshold)
        
        # Find matching segments if plagiarism detected
        if result.is_plagiarized:
            segments = self._find_matching_segments(text1, text2)
            for seg in segments:
                result.add_segment(seg)
        
        return result
    
    def _find_matching_segments(self, text1: str, text2: str) -> List[MatchSegment]:
        """Find specific matching text segments."""
        segments = []
        
        # Split into sentences
        sentences1 = re.split(r'[.!?]+', text1)
        sentences2 = re.split(r'[.!?]+', text2)
        
        pos1 = 0
        for i, sent1 in enumerate(sentences1):
            if len(sent1.strip()) < 20:  # Skip very short sentences
                pos1 += len(sent1) + 1
                continue
            
            pos2 = 0
            for j, sent2 in enumerate(sentences2):
                if len(sent2.strip()) < 20:
                    pos2 += len(sent2) + 1
                    continue
                
                sim = self.calculator.sequence_similarity(sent1, sent2)
                
                if sim >= 70:  # High similarity threshold for segments
                    segment = MatchSegment(
                        sent1.strip(),
                        (pos1, pos1 + len(sent1)),
                        (pos2, pos2 + len(sent2)),
                        sim
                    )
                    segments.append(segment)
                
                pos2 += len(sent2) + 1
            
            pos1 += len(sent1) + 1
        
        return segments[:10]  # Return top 10 matches
    
    def compare_against_corpus(self, target_file: str, 
                               corpus_files: List[str]) -> List[ComparisonResult]:
        """Compare one file against multiple files."""
        results = []
        target_text = self.read_file(target_file)
        
        for corpus_file in corpus_files:
            try:
                result = self.compare_files(target_file, corpus_file)
                results.append(result)
            except Exception as e:
                print(f"Error comparing with {corpus_file}: {str(e)}")
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
    def generate_report(self, results: List[ComparisonResult], 
                       output_file: str, format: str = 'json'):
        """Generate and save a report."""
        if format == 'json':
            report_data = {
                'detection_summary': {
                    'total_comparisons': len(results),
                    'plagiarism_detected': sum(1 for r in results if r.is_plagiarized),
                    'algorithm': self.algorithm,
                    'threshold': self.threshold
                },
                'results': [r.to_dict() for r in results]
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
        
        elif format == 'text':
            with open(output_file, 'w') as f:
                f.write("PLAGIARISM DETECTION BATCH REPORT\n")
                f.write("=" * 60 + "\n\n")
                for result in results:
                    f.write(result.report())
                    f.write("\n")
        
        print(f"Report saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    # Create sample text files for testing
    sample1 = """
    Artificial intelligence is transforming the way we live and work. 
    Machine learning algorithms can now process vast amounts of data 
    and identify patterns that humans might miss. This technology is 
    being applied in healthcare, finance, transportation, and many other fields.
    """
    
    sample2 = """
    AI is revolutionizing our daily lives and professional environments.
    Machine learning algorithms are capable of analyzing huge datasets
    and discovering patterns that may escape human observation. These
    technologies find applications in medical care, banking, transit, and numerous sectors.
    """
    
    sample3 = """
    The weather today is beautiful and sunny. I went for a walk in the park
    and saw many people enjoying the outdoors. It's a perfect day for picnics
    and outdoor activities.
    """
    
    # Create test files
    Path('sample1.txt').write_text(sample1)
    Path('sample2.txt').write_text(sample2)
    Path('sample3.txt').write_text(sample3)
    
    print("=== Plagiarism Detection Tool Demo ===\n")
    
    # Initialize detector
    detector = PlagiarismDetector(algorithm='hybrid', threshold=70.0)
    
    # Compare two similar documents
    print("1. Comparing similar documents...")
    result1 = detector.compare_files('sample1.txt', 'sample2.txt')
    print(result1.report())
    
    # Compare two different documents
    print("\n2. Comparing different documents...")
    result2 = detector.compare_files('sample1.txt', 'sample3.txt')
    print(result2.report())
    
    # Batch comparison
    print("\n3. Batch comparison against corpus...")
    results = detector.compare_against_corpus(
        'sample1.txt',
        ['sample2.txt', 'sample3.txt']
    )
    
    for result in results:
        print(f"\n{result.file2}: {result.similarity_score}% "
              f"({'PLAGIARISM' if result.is_plagiarized else 'ORIGINAL'})")
    
    # Generate reports
    detector.generate_report(results, 'plagiarism_report.json', format='json')
    detector.generate_report(results, 'plagiarism_report.txt', format='text')
    
    print("\n=== Demo Complete ===")
    print("Check 'plagiarism_report.json' and 'plagiarism_report.txt' for detailed reports.")
