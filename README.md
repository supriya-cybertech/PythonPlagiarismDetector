# Plagiarism Detection Tool

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

A powerful Python-based plagiarism detection tool that compares text files and calculates similarity scores using advanced text analysis algorithms. Ideal for educators, content creators, and researchers who need to verify document originality.

![Plagiarism Detection Demo]("docs\Screenshot2026-01-12231208.png")

## ✨ Features

- **Multiple Comparison Algorithms**: Supports various similarity metrics including cosine similarity, Jaccard index, and Levenshtein distance
- **Batch Processing**: Compare multiple documents simultaneously
- **Detailed Reports**: Generate comprehensive similarity reports with highlighted matches
- **Flexible Input**: Supports TXT, DOCX, PDF, and other common text formats
- **Threshold Customization**: Set custom similarity thresholds for plagiarism detection
- **Fast Performance**: Optimized algorithms for quick processing of large documents
- **Visual Output**: Color-coded similarity scores and match visualization

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Algorithms](#algorithms)
- [Output Format](#output-format)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install via pip

```bash
pip install plagiarism-detector
```

### Install from source

```bash
git clone https://github.com/yourusername/plagiarism-detector.git
cd plagiarism-detector
pip install -r requirements.txt
python setup.py install
```

## ⚡ Quick Start

Compare two text files:

```python
from plagiarism_detector import PlagiarismDetector

detector = PlagiarismDetector()
result = detector.compare_files('document1.txt', 'document2.txt')

print(f"Similarity Score: {result.similarity_score}%")
print(f"Plagiarism Detected: {result.is_plagiarized}")
```

## 📖 Usage

### Basic Comparison

```python
from plagiarism_detector import PlagiarismDetector

# Initialize detector
detector = PlagiarismDetector(
    algorithm='cosine',
    threshold=75.0,
    ignore_case=True
)

# Compare two files
result = detector.compare_files('file1.txt', 'file2.txt')

# Access results
print(result.similarity_score)  # Similarity percentage
print(result.matched_segments)  # Specific matching text segments
print(result.report())           # Detailed report
```

### Batch Comparison

```python
# Compare one document against multiple documents
results = detector.compare_against_corpus(
    target_file='submission.txt',
    corpus_files=['doc1.txt', 'doc2.txt', 'doc3.txt']
)

for result in results:
    print(f"{result.file_name}: {result.similarity_score}%")
```

### Advanced Configuration

```python
detector = PlagiarismDetector(
    algorithm='hybrid',           # Use multiple algorithms
    threshold=70.0,               # Custom threshold
    min_match_length=50,          # Minimum words for match
    ignore_case=True,             # Case-insensitive comparison
    ignore_whitespace=True,       # Normalize whitespace
    stemming=True,                # Apply word stemming
    stop_words='english'          # Remove common words
)
```

### Generate Report

```python
# Generate detailed HTML report
detector.generate_report(
    results=results,
    output_file='plagiarism_report.html',
    format='html'
)

# Or JSON format
detector.generate_report(
    results=results,
    output_file='plagiarism_report.json',
    format='json'
)
```

## ⚙️ Configuration

Create a `config.yaml` file for default settings:

```yaml
algorithm: cosine
threshold: 75.0
min_match_length: 30
ignore_case: true
ignore_whitespace: true
stemming: false
stop_words: english
output_format: html
```

Load configuration:

```python
detector = PlagiarismDetector.from_config('config.yaml')
```

## 🧮 Algorithms

### Cosine Similarity
Measures the cosine of the angle between two document vectors. Best for general text comparison.

### Jaccard Index
Calculates the ratio of shared words to total unique words. Effective for shorter texts.

### Levenshtein Distance
Measures character-level differences. Useful for detecting minor modifications.

### Hybrid Approach
Combines multiple algorithms for more accurate detection.

## 📊 Output Format

### Similarity Score
A percentage (0-100%) indicating how similar two documents are.

### Match Details
```json
{
  "similarity_score": 85.5,
  "is_plagiarized": true,
  "algorithm": "cosine",
  "matched_segments": [
    {
      "text": "matching text segment",
      "position_doc1": [100, 150],
      "position_doc2": [200, 250],
      "length": 50
    }
  ],
  "metadata": {
    "file1": "document1.txt",
    "file2": "document2.txt",
    "timestamp": "2025-01-12T10:30:00"
  }
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped improve this tool
- Inspired by academic integrity tools and natural language processing research
- Built with Python and powered by NLTK and scikit-learn

## 📧 Contact

Your Name - [Supriya](Supriya)

Project Link: [https://github.com/supriya-cybertech/PythonPlagiarismDetector.git](https://github.com/supriya-cybertech/PythonPlagiarismDetector.git)

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Keywords**: plagiarism detection, text similarity, document comparison, python, nlp, text analysis, cosine similarity, academic integrity, content verification
