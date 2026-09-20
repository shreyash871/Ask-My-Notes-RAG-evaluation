import re, pathlib, warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

WANTED = ["1", "1A", "7", "8"]

# Longest alternatives first so "1A" wins over "1", "7A" over "7"
HEADING = re.compile(
    r"\bItem\s+(1A|1B|1|7A|7|8|9A|9)\s*[.\-–—:]",
    re.IGNORECASE,
)


def html_to_text(path: pathlib.Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none")):
        tag.decompose()
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_sections(text: str) -> dict[str, str]:
    hits = [(m.group(1).upper(), m.start()) for m in HEADING.finditer(text)]

    # Table-of-contents entries cluster: many headings inside a tiny span.
    # A real section heading has substantial text before the next one.
    MIN_GAP = 2000
    filtered = []
    for i, (item, pos) in enumerate(hits):
        next_pos = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        if next_pos - pos >= MIN_GAP:
            filtered.append((item, pos))

    seen, boundaries = set(), []
    for item, pos in filtered:
        if item not in seen:
            seen.add(item)
            boundaries.append((item, pos))

    boundaries.sort(key=lambda x: x[1])

    sections = {}
    for i, (item, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        if item in WANTED:
            body = text[start:end].strip()
            if len(body) > 1000:
                sections[item] = body
    return sections

if __name__ == "__main__":
    for f in sorted(DATA.glob("*.htm")):
        secs = extract_sections(html_to_text(f))
        found = ", ".join(f"{k}:{len(v):,}" for k, v in sorted(secs.items()))
        print(f"{f.name:22} {found or 'NONE FOUND'}")