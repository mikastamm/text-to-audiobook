class TextUtils:
    ABBREVS = {"dr", "mr", "mrs", "ms", "etc"}

    @staticmethod
    def _split_regular_text(text: str):
        sentences = []
        buffer = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            buffer.append(ch)
            if ch in {'.', '!', '?'}:
                next_char = text[i + 1] if (i + 1 < n) else ''
                if next_char.isspace() or i == n - 1:
                    temp_str = "".join(buffer)
                    import re
                    match = re.search(r"\b([A-Za-z]+)\.$", temp_str)
                    if match:
                        word_before_dot = match.group(1).lower()
                        if word_before_dot in TextUtils.ABBREVS:
                            pass
                        else:
                            sentences.append(temp_str.strip())
                            buffer = []
                            while i + 1 < n and text[i + 1].isspace():
                                i += 1
                    else:
                        sentences.append(temp_str.strip())
                        buffer = []
                        while i + 1 < n and text[i + 1].isspace():
                            i += 1
            i += 1
        leftover = "".join(buffer).strip()
        if leftover:
            sentences.append(leftover)
        return sentences

    @staticmethod
    def initial_split_into_sentences(text: str):
        import re
        text = re.sub(r'\s*</pause>', '', text)
        parts = re.split(r'(<pause\s+duration="(?:short|long)"\s*/?>)', text)
        sentences = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                sub_sentences = TextUtils._split_regular_text(part)
                if sub_sentences:
                    sentences.extend(sub_sentences)
            else:
                m = re.search(r'duration="(short|long)"', part)
                duration = m.group(1) if m else "short"
                if sentences:
                    sentences[-1] = sentences[-1].strip() + " ||PAUSE:" + duration + "||"
                else:
                    sentences.append("||PAUSE:" + duration + "||")
        return sentences

    @staticmethod
    def force_split_long_text(text: str, max_chars: int):
        if len(text) <= max_chars:
            return [text]
        splits = []
        start = 0
        length = len(text)
        while start < length:
            end = start + max_chars
            if end >= length:
                splits.append(text[start:].strip())
                break
            space_index = text.rfind(" ", start, end)
            if space_index == -1 or space_index < start:
                splits.append(text[start:end].strip())
                start = end
            else:
                splits.append(text[start:space_index].strip())
                start = space_index + 1
        return splits

    @staticmethod
    def split_on_delims(long_text: str, max_len: int):
        def splitter(text, delimiters):
            pieces = [text]
            for delim in delimiters:
                new_pieces = []
                for p in pieces:
                    if len(p) <= max_len:
                        new_pieces.append(p)
                    else:
                        sub_parts = p.split(delim)
                        for sp in sub_parts:
                            sp = sp.strip()
                            if sp:
                                new_pieces.append(sp)
                pieces = new_pieces
            return pieces
        candidate_delims = [" - ", ", "]
        splitted = splitter(long_text, candidate_delims)
        final_list = []
        for piece in splitted:
            if len(piece) > max_len:
                final_list.extend(TextUtils.force_split_long_text(piece, max_chars=max_len))
            else:
                final_list.append(piece)
        return final_list

    @staticmethod
    def combine_into_blocks(sentences, min_chars: int, max_chars: int):
        import re
        blocks = []
        current = ""
        for sentence in sentences:
            m = re.search(r'\|\|PAUSE:(short|long)\|\|$', sentence)
            if m:
                pause = m.group(1)
                sentence_text = sentence[:sentence.rfind("||PAUSE:")].strip()
                if current:
                    blocks.append((current.strip(), ""))
                    current = ""
                blocks.append((sentence_text, pause))
            else:
                if not current:
                    current = sentence
                else:
                    if len(current) + 1 + len(sentence) <= max_chars:
                        current += " " + sentence
                    else:
                        if len(current) >= min_chars:
                            blocks.append((current.strip(), ""))
                            current = sentence
                        else:
                            current += " " + sentence
        if current.strip():
            blocks.append((current.strip(), ""))
        return blocks

    @staticmethod
    def chunk_text(text: str, max_chars: int = 10000):
        sentences = TextUtils.initial_split_into_sentences(text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + 1 + len(sentence) <= max_chars:
                if current:
                    current += " " + sentence
                else:
                    current = sentence
            else:
                if current:
                    chunks.append(current.strip())
                if len(sentence) > max_chars:
                    chunks.extend(TextUtils.force_split_long_text(sentence, max_chars=max_chars))
                    current = ""
                else:
                    current = sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks
