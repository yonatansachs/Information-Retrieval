TOKENIZATION_RULES = {
    'min_length': 2,  # Minimum token length
    'keep_numbers': True,  # Whether to keep numeric tokens
    'keep_emails': True,  # Whether to keep email addresses as single tokens
    'keep_urls': True,  # Whether to keep URLs as single tokens
    'keep_hyphens': True,  # Whether to keep hyphenated words together
    'keep_underscores': True,  # Whether to keep words with underscores
    'lowercase': True,  # Whether to convert to lowercase
    'remove_punctuation': True,  # Whether to remove punctuation
    'special_cases': [  # Special cases to keep as single tokens
        'c++',
        'c#',
        'f#',
        '.net',
        'u.s.a',
        'u.k'
    ]
}