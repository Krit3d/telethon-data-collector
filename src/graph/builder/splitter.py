import re

from pydantic import BaseModel


class TextChunk(BaseModel):
    chunk_index: int
    total_chunks: int
    text: str
    is_chunked: bool


class TextSplitter:
    _BOT_DISCLAIMER_RE = re.compile(
        r"(?i)(?:^|\n)\s*(?:@\w+|t\.me/\w+)[^\n]*"
        r"(?:bot|бот)[^\n]*\n?",
    )
    _URL_RE = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+")
    _MARKUP_RE = re.compile(r"[*_~`]")
    _MULTI_WS_RE = re.compile(r"[ \t]+")
    _MULTI_NL_RE = re.compile(r"\n{3,}")
    _PARAGRAPH_RE = re.compile(r"\n\s*\n")

    @staticmethod
    def sanitize_text(text: str | None) -> str:
        if not text:
            return ""
        cleaned = TextSplitter._BOT_DISCLAIMER_RE.sub("", text)
        cleaned = TextSplitter._URL_RE.sub("", cleaned)
        cleaned = TextSplitter._MARKUP_RE.sub("", cleaned)
        cleaned = TextSplitter._MULTI_WS_RE.sub(" ", cleaned)
        cleaned = TextSplitter._MULTI_NL_RE.sub("\n\n", cleaned)
        return cleaned.strip()

    def prepare_and_split(
        self,
        content: str | None,
        transcription: str | None,
        max_chars: int = 4000,
        overlap: int = 200,
    ) -> list[TextChunk]:
        clean_content = self.sanitize_text(content)
        clean_transcription = self.sanitize_text(transcription)

        if clean_content and clean_transcription:
            combined = f"{clean_content}\n\n[Audio Transcription]:\n{clean_transcription}"
        elif clean_content:
            combined = clean_content
        elif clean_transcription:
            combined = clean_transcription
        else:
            combined = ""

        if not combined:
            return [TextChunk(chunk_index=0, total_chunks=1, text="", is_chunked=False)]

        if len(combined) <= max_chars:
            return [TextChunk(chunk_index=0, total_chunks=1, text=combined, is_chunked=False)]

        chunks = self._split_by_paragraphs(combined, max_chars, overlap)
        total = len(chunks)
        return [
            TextChunk(chunk_index=i, total_chunks=total, text=chunk, is_chunked=True)
            for i, chunk in enumerate(chunks)
        ]

    def _split_by_paragraphs(self, text: str, max_chars: int, overlap: int) -> list[str]:
        paragraphs = [p.strip() for p in self._PARAGRAPH_RE.split(text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if not current:
                current = para
                continue
            if len(current) + 1 + len(para) <= max_chars:
                current = f"{current}\n\n{para}"
                continue
            chunks.append(current)
            current = para

        if current:
            chunks.append(current)

        merged: list[str] = []
        for chunk in chunks:
            if len(chunk) > max_chars:
                merged.extend(self._split_long(chunk, max_chars, overlap))
            else:
                merged.append(chunk)

        if len(merged) > 1 and overlap > 0:
            merged = self._apply_overlap(merged, overlap)

        return merged

    def _split_long(self, text: str, max_chars: int, overlap: int) -> list[str]:
        parts: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + max_chars, length)
            if end < length:
                nl_boundary = text.rfind("\n", start, end)
                space_boundary = text.rfind(" ", start, end)
                boundary = max(nl_boundary, space_boundary)
                if boundary > start + max_chars // 2:
                    end = boundary
            parts.append(text[start:end].strip())
            if end >= length:
                break
            start = max(end - overlap, start + 1)
        return [p for p in parts if p]

    def _apply_overlap(self, chunks: list[str], overlap: int) -> list[str]:
        result: list[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
                continue
            prev = result[-1]
            tail = self._overlap_tail(prev, overlap)
            result.append(f"{tail}\n{chunk}")
        return result

    def _overlap_tail(self, text: str, overlap: int) -> str:
        if len(text) <= overlap:
            return text
        start = len(text) - overlap
        boundary = max(
            text.rfind(" ", 0, start),
            text.rfind("\n", 0, start),
            text.rfind("\t", 0, start),
            text.rfind("\r", 0, start),
        )
        if boundary > 0:
            start = boundary + 1
        return text[start:].strip()
