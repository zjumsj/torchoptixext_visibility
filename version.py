import os
import glob

class Version:
    """
    A class for comparing version strings with arbitrary formats

    Features:
    - Handles numeric versions (1.2.3)
    - Handles alphanumeric versions (1.0.1a, 2.0-beta)
    - Handles versions of different lengths (1.2 vs 1.2.0)
    - Supports all comparison operators: ==, !=, <, <=, >, >=
    """

    def __init__(self, version_str: str):
        """
        Initialize a Version object

        Args:
            version_str: Version string (e.g., "7.2.0", "1.0.1a")
        """
        self.original = version_str
        self.parts = self._parse_version(version_str)

    def _parse_version(self, version_str: str) -> list:
        """
        Parse version string into comparable components

        Rules:
        1. Split by dot (.) separator
        2. Convert pure numeric parts to integers
        3. Handle alphanumeric parts by separating digits and letters
        4. Convert letter parts to lowercase for case-insensitive comparison

        Args:
            version_str: Input version string

        Returns:
            List of version components (integers and strings)
        """
        parts = []
        for part in version_str.split('.'):
            # Handle pure numeric parts
            if part.isdigit():
                parts.append(int(part))
            # Handle alphanumeric parts
            else:
                # Separate numeric prefix and alphabetic suffix
                num_part = ''
                str_part = ''
                for char in part:
                    if char.isdigit():
                        num_part += char
                    else:
                        str_part += char.lower()  # Case-insensitive comparison

                # Add numeric part if exists
                if num_part:
                    parts.append(int(num_part))
                # Add string part if exists
                if str_part:
                    parts.append(str_part)
        return parts

    def _normalize_parts(self, other_parts: list) -> tuple:
        """
        Normalize two version part lists to the same length

        Rules:
        - Pad shorter list with zeros (for numeric) or empty strings (for text)
        - Ensure comparable types at each position

        Args:
            other_parts: Parts list from another Version object

        Returns:
            Tuple of normalized part lists
        """
        max_len = max(len(self.parts), len(other_parts))
        a_parts = self.parts + [0] * (max_len - len(self.parts))
        b_parts = other_parts + [0] * (max_len - len(other_parts))

        # Ensure comparable types at each position
        normalized_a = []
        normalized_b = []
        for a, b in zip(a_parts, b_parts):
            if type(a) == type(b):
                normalized_a.append(a)
                normalized_b.append(b)
            else:
                # Convert to strings for mixed-type comparison
                normalized_a.append(str(a))
                normalized_b.append(str(b))

        return normalized_a, normalized_b

    def __eq__(self, other: 'Version') -> bool:
        """Equality operator (==)"""
        if not isinstance(other, Version):
            return False
        a, b = self._normalize_parts(other.parts)
        return a == b

    def __ne__(self, other: 'Version') -> bool:
        """Inequality operator (!=)"""
        return not self.__eq__(other)

    def __lt__(self, other: 'Version') -> bool:
        """Less than operator (<)"""
        a, b = self._normalize_parts(other.parts)
        for i in range(len(a)):
            # Numeric comparison
            if isinstance(a[i], int) and isinstance(b[i], int):
                if a[i] < b[i]:
                    return True
                if a[i] > b[i]:
                    return False
            # String comparison
            elif isinstance(a[i], str) and isinstance(b[i], str):
                if a[i] < b[i]:
                    return True
                if a[i] > b[i]:
                    return False
            # Mixed-type comparison
            else:
                a_str = str(a[i])
                b_str = str(b[i])
                if a_str < b_str:
                    return True
                if a_str > b_str:
                    return False
        return False

    def __le__(self, other: 'Version') -> bool:
        """Less than or equal operator (<=)"""
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: 'Version') -> bool:
        """Greater than operator (>)"""
        return not self.__le__(other)

    def __ge__(self, other: 'Version') -> bool:
        """Greater than or equal operator (>=)"""
        return not self.__lt__(other)

    def __str__(self) -> str:
        """String representation"""
        return self.original

    def __repr__(self) -> str:
        """Object representation"""
        return f"Version('{self.original}')"

def replace_blackslashes(s):
    return s.replace('\\','/')

def strip_quotes(s):
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s

def fill_template(src_filename, tar_filename, replace_dict):
    # 1. Read content
    with open(src_filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. Safe replacement processing (avoids nested replacement issues)
    # Sort keys by length descending to prioritize longer keys
    sorted_keys = sorted(replace_dict.keys(), key=len, reverse=True)

    # 3. Execute replacements
    for key in sorted_keys:
        # Ensure key and value are strings
        str_key = str(key)
        str_value = str(replace_dict[key])
        content = content.replace(str_key, str_value)

    # 4. Write to target file (using UTF-8 encoding)
    with open(tar_filename, 'w', encoding='utf-8') as f:
        f.write(content)

def find_nvrtc_dll(cuda_path: str) -> str:
    """
    Locates the nvrtc64*.dll file within the CUDA installation directory.

    This function searches common CUDA library paths to find the latest version
    of the NVIDIA Runtime Compiler (NVRTC) dynamic link library.

    Parameters:
        cuda_path (str): Path to the CUDA Toolkit installation directory

    Returns:
        str: Full path to the located nvrtc64*.dll file

    Raises:
        FileNotFoundError: If no matching DLL file is found in any search paths
        ValueError: If the provided cuda_path is invalid or inaccessible

    Search Strategy:
        1. Checks common CUDA library directories in priority order
        2. Uses glob pattern matching to find nvrtc64*.dll files
        3. Selects the highest version number when multiple matches exist
    """
    # Validate CUDA installation path
    if not os.path.exists(cuda_path):
        raise ValueError(f"Invalid CUDA path: '{cuda_path}' does not exist")

    # Define priority search paths for CUDA libraries
    search_paths = [
        os.path.join(cuda_path, "bin"),  # Primary location for DLLs
        os.path.join(cuda_path, "lib", "x64"),  # 64-bit Windows libraries
        os.path.join(cuda_path, "lib64"),  # Linux-style library path
        os.path.join(cuda_path, "lib")  # Generic library path
    ]

    # Traverse through all potential library locations
    for path in search_paths:
        # Skip non-existent paths
        if not os.path.exists(path):
            continue

        # Search for nvrtc64 DLL files using pattern matching
        pattern = os.path.join(path, "nvrtc64_*.dll")
        matches = glob.glob(pattern)

        # Process found matches
        if matches:
            # Sort matches by filename to get highest version (reverse sort)
            matches.sort(reverse=True)
            return matches[0]  # Return the first (highest version) match

    # Generate detailed error message if no matches found
    searched_paths = "\n".join(f"- {p}" for p in search_paths)
    raise FileNotFoundError(
        f"No nvrtc64_*.dll files found in CUDA installation at: {cuda_path}\n"
        f"The following locations were searched:\n{searched_paths}\n"
        "Please verify your CUDA installation or provide the correct path."
    )

# Test function to validate implementation
def test_version_comparison():
    """Test version comparison functionality"""
    test_cases = [
        # Equality tests (both is_less and is_greater should be False for equal versions)
        ("1.0", "1.0", False, False),
        ("2.0.0", "2.0", False, False),
        ("3.0.0", "3.0.0", False, False),
        ("1.0a", "1.0a", False, False),

        # Less than tests (is_less = True, is_greater = False)
        ("1.0", "2.0", True, False),
        ("1.0.1", "1.1", True, False),
        ("1.9", "1.10", True, False),
        ("1.0a", "1.0b", True, False),
        ("1.0-beta", "1.0", True, False),

        # Greater than tests (is_less = False, is_greater = True)
        ("2.0", "1.0", False, True),
        ("1.2", "1.1.9", False, True),
        ("1.10", "1.9", False, True),
        ("1.0b", "1.0a", False, True),
        ("1.0", "1.0-beta", False, True),

        # Complex cases
        ("10.1", "9.9.9", False, True),
        ("1.0.0.1", "1.0", False, True),
        ("1.0.1", "1.0.0.9", False, True),
        ("2.0.0-alpha", "2.0.0-beta", True, False),
        ("2.0.0-rc1", "2.0.0-rc2", True, False),
    ]

    for v1_str, v2_str, is_less, is_greater in test_cases:
        v1 = Version(v1_str)
        v2 = Version(v2_str)

        # Test less than
        if is_less:
            assert v1 < v2, f"Error: {v1} should be less than {v2}"
        else:
            assert not (v1 < v2), f"Error: {v1} should not be less than {v2}"

        # Test greater than
        if is_greater:
            assert v1 > v2, f"Error: {v1} should be greater than {v2}"
        else:
            assert not (v1 > v2), f"Error: {v1} should not be greater than {v2}"

        # Test equality
        if not is_less and not is_greater:
            assert v1 == v2, f"Error: {v1} should equal {v2}"
        else:
            assert v1 != v2, f"Error: {v1} should not equal {v2}"

    print("All tests passed!")


# Example usage
if __name__ == "__main__":
    # Run test cases
    test_version_comparison()

    # Demonstration
    v1 = Version("7.2.0")
    v2 = Version("8.0")

    print(f"\nDemonstration:")
    print(f"{v1} < {v2}: {v1 < v2}")  # True
    print(f"{v1} > {v2}: {v1 > v2}")  # False
    print(f"{v1} == {v2}: {v1 == v2}")  # False

    v3 = Version("1.0.1a")
    v4 = Version("1.0.1b")
    print(f"\n{v3} < {v4}: {v3 < v4}")  # True

    v5 = Version("2.0.0")
    v6 = Version("2.0.0-rc1")
    print(f"{v5} > {v6}: {v5 > v6}")  # True
    print(f"{v6} < {v5}: {v6 < v5}")  # True