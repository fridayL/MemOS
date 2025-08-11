def split_continuous_references(text: str) -> str:
    """
    Split continuous reference tags into individual reference tags.

    Converts patterns like [1:92ff35fb, 4:bfe6f044] to [1:92ff35fb] [4:bfe6f044]

    Only processes text if:
    1. '[' appears exactly once
    2. ']' appears exactly once
    3. Contains commas between '[' and ']'

    Args:
        text (str): Text containing reference tags

    Returns:
        str: Text with split reference tags, or original text if conditions not met
    """
    # Early return if text is empty
    if not text:
        return text
    # Check if '[' appears exactly once
    if text.count("[") != 1:
        return text
    # Check if ']' appears exactly once
    if text.count("]") != 1:
        return text
    # Find positions of brackets
    open_bracket_pos = text.find("[")
    close_bracket_pos = text.find("]")

    # Check if brackets are in correct order
    if open_bracket_pos >= close_bracket_pos:
        return text
    # Extract content between brackets
    content_between_brackets = text[open_bracket_pos + 1 : close_bracket_pos]
    # Check if there's a comma between brackets
    if "," not in content_between_brackets:
        return text
    text = text.replace(content_between_brackets, content_between_brackets.replace(", ", "]["))
    text = text.replace(content_between_brackets, content_between_brackets.replace(",", "]["))

    return text


def process_streaming_references_complete(text_buffer: str) -> tuple[str, str]:
    """
    Complete streaming reference processing to ensure reference tags are never split.

    Enhanced to support multiple reference formats:
    - [1:xxxx] (standard format)
    - [refid:xxxx] (alternative format)
    - [memid:xxxx], [ref:xxxx], [mem:xxxx] (other formats)

    Args:
        text_buffer (str): The accumulated text buffer.

    Returns:
        tuple[str, str]: (processed_text, remaining_buffer)
    """
    import re

    # Enhanced patterns for different reference formats
    # Standard format: [1:memoriesID] or [2:abc123]
    standard_pattern = r"\[\d+:[^\]]+\]"

    # Alternative formats: [refid:memoriesID], [ref:abc], [memid:xyz], etc.
    alt_pattern = r"\[(?:ref|refid|memid|mem):[^\]]+\]"

    # Combined complete pattern
    complete_pattern = f"({standard_pattern}|{alt_pattern})"

    # Find all complete reference tags
    complete_matches = list(re.finditer(complete_pattern, text_buffer))

    if complete_matches:
        # Find the last complete tag
        last_match = complete_matches[-1]
        end_pos = last_match.end()

        # Check if there's any incomplete reference after the last complete one
        remaining_text = text_buffer[end_pos:]

        # Look for potential incomplete reference patterns after the last complete tag
        # This includes partial standard and alternative formats
        incomplete_patterns = [
            r"\[\d*:?[^\]]*$",  # Standard format incomplete
            r"\[(?:ref|refid|memid|mem):?[^\]]*$",  # Alternative format incomplete
        ]

        for incomplete_pattern in incomplete_patterns:
            if re.search(incomplete_pattern, remaining_text):
                # There's a potential incomplete reference, find where it starts
                incomplete_match = re.search(incomplete_pattern, remaining_text)
                if incomplete_match:
                    incomplete_start = end_pos + incomplete_match.start()
                    processed_text = text_buffer[:incomplete_start]
                    remaining_buffer = text_buffer[incomplete_start:]

                    # Apply reference splitting to the processed text
                    processed_text = split_continuous_references(processed_text)
                    return processed_text, remaining_buffer

        # No incomplete reference after the last complete tag, process all
        processed_text = split_continuous_references(text_buffer)
        return processed_text, ""

    # Check for incomplete reference tags - handle multiple formats
    opening_patterns = [
        r"\[\d+:",  # Standard: [1:, [22:, etc.
        r"\[(?:ref|refid|memid|mem):",  # Alternative: [refid:, [ref:, etc.
    ]

    for opening_pattern in opening_patterns:
        opening_matches = list(re.finditer(opening_pattern, text_buffer))

        if opening_matches:
            # Find the last opening tag
            last_opening = opening_matches[-1]
            opening_start = last_opening.start()

            # Check if this might be a complete reference tag (has closing bracket after the pattern)
            remaining_text = text_buffer[last_opening.end() :]
            if "]" in remaining_text:
                # This looks like a complete reference tag, process it
                processed_text = split_continuous_references(text_buffer)
                return processed_text, ""
            else:
                # Incomplete reference tag, keep it in buffer
                processed_text = text_buffer[:opening_start]
                processed_text = split_continuous_references(processed_text)
                return processed_text, text_buffer[opening_start:]

    # Enhanced check for potential reference patterns
    # Handle various partial reference starts including alternative formats
    potential_ref_patterns = [
        r"\[\d*:?$",  # Standard: [, [1, [12:, etc. at end of buffer
        r"\[(?:ref|refid|memid|mem):?$",  # Alternative: [ref, [refid:, etc. at end of buffer
        r"\[r$",  # Partial [r
        r"\[re$",  # Partial [re
        r"\[ref$",  # Partial [ref
        r"\[refi$",  # Partial [refi
        r"\[m$",  # Partial [m
        r"\[me$",  # Partial [me
        r"\[mem$",  # Partial [mem
        r"\[memi$",  # Partial [memi
    ]

    for potential_ref_pattern in potential_ref_patterns:
        if re.search(potential_ref_pattern, text_buffer):
            # Find the position of the potential reference start
            match = re.search(potential_ref_pattern, text_buffer)
            if match:
                ref_start = match.start()
                processed_text = text_buffer[:ref_start]
                processed_text = split_continuous_references(processed_text)
                return processed_text, text_buffer[ref_start:]

    # Check for standalone [ only at the very end of the buffer
    # This prevents cutting off mathematical expressions like [ \Delta U = Q - W ]
    # But we need to be more careful about what constitutes a potential reference
    if text_buffer.endswith("["):
        # Look ahead in context to see if this might be a reference
        # If the previous characters suggest it's not a mathematical expression, hold it back
        if len(text_buffer) >= 2:
            # Check the character before [
            prev_char = text_buffer[-2]
            # If it's a space or punctuation, it might be a reference start
            if prev_char in " \n\t.,;:!?":
                processed_text = text_buffer[:-1]
                processed_text = split_continuous_references(processed_text)
                return processed_text, "["
        else:
            # If buffer is just "[", hold it back
            processed_text = text_buffer[:-1]
            processed_text = split_continuous_references(processed_text)
            return processed_text, "["

    # No reference-like patterns found, process all text
    processed_text = split_continuous_references(text_buffer)
    return processed_text, ""


def process_streaming_references_enhanced(text_buffer: str) -> tuple[str, str]:
    """
    Enhanced streaming reference processing to handle multiple reference formats.

    Handles formats like:
    - [1:xxxx] (standard)
    - [refid:xxxx] (alternative format)
    - [2:outmemory] (non-standard content)

    Avoids processing mathematical expressions like [x+y], [a-b], etc.

    Args:
        text_buffer (str): The accumulated text buffer.

    Returns:
        tuple[str, str]: (processed_text, remaining_buffer)
    """
    import re

    # Extended patterns for different reference formats
    # Standard format: [1:memoriesID] or [2:abc123]
    standard_pattern = r"\[\d+:[^\]]+\]"

    # Alternative formats: [refid:memoriesID], [ref:abc], [memid:xyz], etc.
    alt_pattern = r"\[(?:ref|refid|memid|mem):[^\]]+\]"

    # Combined complete pattern
    complete_pattern = f"({standard_pattern}|{alt_pattern})"

    # Find all complete reference tags
    complete_matches = list(re.finditer(complete_pattern, text_buffer))

    if complete_matches:
        # Find the last complete tag
        last_match = complete_matches[-1]
        end_pos = last_match.end()

        # Check if there's any incomplete reference after the last complete one
        remaining_text = text_buffer[end_pos:]

        # Look for potential incomplete reference patterns after the last complete tag
        # This includes partial standard and alternative formats
        incomplete_patterns = [
            r"\[\d*:?[^\]]*$",  # Standard format incomplete
            r"\[(?:ref|refid|memid|mem):?[^\]]*$",  # Alternative format incomplete
        ]

        for incomplete_pattern in incomplete_patterns:
            if re.search(incomplete_pattern, remaining_text):
                # There's a potential incomplete reference, find where it starts
                incomplete_match = re.search(incomplete_pattern, remaining_text)
                if incomplete_match:
                    incomplete_start = end_pos + incomplete_match.start()
                    processed_text = text_buffer[:incomplete_start]
                    remaining_buffer = text_buffer[incomplete_start:]

                    # Apply reference splitting to the processed text
                    processed_text = split_continuous_references_enhanced(processed_text)
                    return processed_text, remaining_buffer

        # No incomplete reference after the last complete tag, process all
        processed_text = split_continuous_references_enhanced(text_buffer)
        return processed_text, ""

    # Check for incomplete reference tags - handle multiple formats
    opening_patterns = [
        r"\[\d+:",  # Standard: [1:, [22:, etc.
        r"\[(?:ref|refid|memid|mem):",  # Alternative: [refid:, [ref:, etc.
    ]

    for opening_pattern in opening_patterns:
        opening_matches = list(re.finditer(opening_pattern, text_buffer))

        if opening_matches:
            # Find the last opening tag
            last_opening = opening_matches[-1]
            opening_start = last_opening.start()

            # Check if this might be a complete reference tag (has closing bracket after the pattern)
            remaining_text = text_buffer[last_opening.end() :]
            if "]" in remaining_text:
                # This looks like a complete reference tag, process it
                processed_text = split_continuous_references_enhanced(text_buffer)
                return processed_text, ""
            else:
                # Incomplete reference tag, keep it in buffer
                processed_text = text_buffer[:opening_start]
                processed_text = split_continuous_references_enhanced(processed_text)
                return processed_text, text_buffer[opening_start:]

    # More sophisticated check for potential reference patterns
    # Handle various partial reference starts
    potential_ref_patterns = [
        r"\[\d*:?$",  # Standard: [, [1, [12:, etc. at end of buffer
        r"\[(?:ref|refid|memid|mem):?$",  # Alternative: [ref, [refid:, etc. at end of buffer
        r"\[r$",  # Partial [r
        r"\[re$",  # Partial [re
        r"\[ref$",  # Partial [ref
        r"\[refi$",  # Partial [refi
        r"\[m$",  # Partial [m
        r"\[me$",  # Partial [me
        r"\[mem$",  # Partial [mem
        r"\[memi$",  # Partial [memi
    ]

    for potential_ref_pattern in potential_ref_patterns:
        if re.search(potential_ref_pattern, text_buffer):
            # Find the position of the potential reference start
            match = re.search(potential_ref_pattern, text_buffer)
            if match:
                ref_start = match.start()
                processed_text = text_buffer[:ref_start]
                processed_text = split_continuous_references_enhanced(processed_text)
                return processed_text, text_buffer[ref_start:]

    # Check for standalone [ only at the very end of the buffer
    # This prevents cutting off mathematical expressions like [ \Delta U = Q - W ]
    # But we need to be more careful about what constitutes a potential reference
    if text_buffer.endswith("["):
        # Look ahead in context to see if this might be a reference
        # If the previous characters suggest it's not a mathematical expression, hold it back
        if len(text_buffer) >= 2:
            # Check the character before [
            prev_char = text_buffer[-2]
            # If it's a space or punctuation, it might be a reference start
            if prev_char in " \n\t.,;:!?":
                processed_text = text_buffer[:-1]
                processed_text = split_continuous_references_enhanced(processed_text)
                return processed_text, "["
        else:
            # If buffer is just "[", hold it back
            processed_text = text_buffer[:-1]
            processed_text = split_continuous_references_enhanced(processed_text)
            return processed_text, "["

    # No reference-like patterns found, process all text
    processed_text = split_continuous_references_enhanced(text_buffer)
    return processed_text, ""


def split_continuous_references_enhanced(text: str) -> str:
    """
    Enhanced version of split_continuous_references that handles multiple reference formats.

    Converts patterns like:
    - [1:92ff35fb, 4:bfe6f044] to [1:92ff35fb] [4:bfe6f044]
    - [refid:abc123, refid:def456] to [refid:abc123] [refid:def456]
    - Mixed formats: [1:abc, refid:def] to [1:abc] [refid:def]

    Args:
        text (str): Text containing reference tags

    Returns:
        str: Text with split reference tags, or original text if conditions not met
    """
    # Early return if text is empty
    if not text:
        return text

    # Check if '[' appears exactly once
    if text.count("[") != 1:
        return text

    # Check if ']' appears exactly once
    if text.count("]") != 1:
        return text

    # Find positions of brackets
    open_bracket_pos = text.find("[")
    close_bracket_pos = text.find("]")

    # Check if brackets are in correct order
    if open_bracket_pos >= close_bracket_pos:
        return text

    # Extract content between brackets
    content_between_brackets = text[open_bracket_pos + 1 : close_bracket_pos]

    # Check if there's a comma between brackets
    if "," not in content_between_brackets:
        return text

    # Check if this looks like reference content (avoid mathematical expressions)
    # Reference content should contain colons and alphanumeric characters
    import re

    reference_pattern = r"(?:\d+:[^\s,]+|(?:ref|refid|memid|mem):[^\s,]+)"

    # Split by comma and check if each part looks like a reference
    parts = [part.strip() for part in content_between_brackets.split(",")]
    valid_parts = []

    for part in parts:
        if re.match(reference_pattern, part):
            valid_parts.append(part)
        else:
            # If any part doesn't look like a reference, don't process
            return text

    # If all parts look like references, proceed with splitting
    if len(valid_parts) == len(parts):
        # Replace with proper spacing
        new_content = content_between_brackets.replace(", ", "][")
        new_content = new_content.replace(",", "][")
        text = text.replace(content_between_brackets, new_content)

    return text
