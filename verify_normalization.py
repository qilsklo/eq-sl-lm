from urllib.parse import urlparse

def normalize_url(url):
    try:
        parsed = urlparse(url)
        # Lowercase scheme and netloc
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path
        
        # Remove trailing slash from path
        if path and path.endswith('/'):
            path = path[:-1]
            
        # Reconstruct without fragment
        normalized = f"{scheme}://{netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
            
        return normalized
    except:
        return url

test_cases = [
    ("HTTP://Example.com", "http://example.com"),
    ("http://example.com/", "http://example.com"),
    ("http://example.com/foo/", "http://example.com/foo"),
    ("http://example.com/foo#bar", "http://example.com/foo"),
    ("https://Example.com/Foo/Bar/", "https://example.com/Foo/Bar"), # Path case preserved? My logic preserves path case.
    ("http://example.com/?q=1", "http://example.com?q=1"),
]

print("Verifying URL Normalization...")
all_passed = True
for input_url, expected in test_cases:
    result = normalize_url(input_url)
    if result == expected:
        print(f"[PASS] {input_url} -> {result}")
    else:
        print(f"[FAIL] {input_url} -> {result} (Expected: {expected})")
        all_passed = False

if all_passed:
    print("\nAll tests passed!")
else:
    print("\nSome tests failed.")
