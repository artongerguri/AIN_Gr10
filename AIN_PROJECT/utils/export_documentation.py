from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "DOKUMENTIM_PROJEKTI.md"
HTML_OUT = ROOT / "DOKUMENTIM_PROJEKTI_export.html"
DOCX_OUT = ROOT / "DOKUMENTIM_PROJEKTI.docx"
DOCUMENTS = [
    {
        "source": ROOT / "DOKUMENTIM_PROJEKTI.md",
        "html": ROOT / "DOKUMENTIM_PROJEKTI_export.html",
        "docx": ROOT / "DOKUMENTIM_PROJEKTI.docx",
        "title": "Dokumentimi i Projektit",
    },
    {
        "source": ROOT / "RRJEDHA_EKZEKUTIMIT_TOY.md",
        "html": ROOT / "RRJEDHA_EKZEKUTIMIT_TOY_export.html",
        "docx": ROOT / "RRJEDHA_EKZEKUTIMIT_TOY.docx",
        "title": "Rrjedha e Ekzekutimit - Toy Instance",
    },
]


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    items: list[str] | None = None
    rows: list[list[str]] | None = None


def parse_markdown(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            fence = stripped[:3]
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            blocks.append(Block("code", text="\n".join(code_lines)))
            continue

        if re.match(r"^#{1,6}\s+", stripped):
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(Block("heading", text=stripped[level:].strip(), level=level))
            i += 1
            continue

        if stripped in {"---", "***", "___"}:
            blocks.append(Block("hr"))
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                rows.append(row)
                i += 1
            if len(rows) >= 2 and all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in rows[1]):
                rows.pop(1)
            blocks.append(Block("table", rows=rows))
            continue

        if re.match(r"^[-*]\s+", stripped):
            items: list[str] = []
            while i < len(lines):
                current = lines[i].strip()
                match = re.match(r"^[-*]\s+(.*)$", current)
                if not match:
                    break
                items.append(match.group(1).strip())
                i += 1
            blocks.append(Block("list", items=items))
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if nxt.startswith("```") or re.match(r"^#{1,6}\s+", nxt) or nxt in {"---", "***", "___"}:
                break
            if nxt.startswith("|") or re.match(r"^[-*]\s+", nxt):
                break
            para_lines.append(nxt)
            i += 1
        blocks.append(Block("paragraph", text=" ".join(para_lines)))

    return blocks


def inline_html(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{html_escape(part[1:-1])}</code>")
        else:
            out.append(html_escape(part))
    return "".join(out)


def build_html(blocks: list[Block], title: str = "Dokumentim Profesional i Projektit") -> str:
    body: list[str] = []
    for block in blocks:
        if block.kind == "heading":
            level = min(max(block.level, 1), 4)
            body.append(f"<h{level}>{inline_html(block.text)}</h{level}>")
        elif block.kind == "paragraph":
            body.append(f"<p>{inline_html(block.text)}</p>")
        elif block.kind == "list":
            body.append("<ul>")
            for item in block.items or []:
                body.append(f"<li>{inline_html(item)}</li>")
            body.append("</ul>")
        elif block.kind == "code":
            body.append(f"<pre><code>{html_escape(block.text)}</code></pre>")
        elif block.kind == "table":
            rows = block.rows or []
            body.append("<table>")
            for ridx, row in enumerate(rows):
                tag = "th" if ridx == 0 else "td"
                body.append("<tr>" + "".join(f"<{tag}>{inline_html(cell)}</{tag}>" for cell in row) + "</tr>")
            body.append("</table>")
        elif block.kind == "hr":
            body.append("<hr>")

    return f"""<!doctype html>
<html lang="sq">
<head>
  <meta charset="utf-8">
  <title>{html_escape(title)}</title>
  <style>
    @page {{ size: A4; margin: 18mm 17mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: #111827;
      font-family: Arial, "Segoe UI", sans-serif;
      font-size: 11.5pt;
      line-height: 1.48;
    }}
    h1, h2, h3, h4 {{
      color: #0f172a;
      break-after: avoid;
      margin: 20px 0 8px;
      line-height: 1.22;
    }}
    h1 {{ font-size: 26pt; margin-top: 0; }}
    h2 {{ font-size: 18pt; border-bottom: 1px solid #d9dee7; padding-bottom: 5px; }}
    h3 {{ font-size: 14pt; }}
    h4 {{ font-size: 12.5pt; }}
    p {{ margin: 0 0 9px; }}
    ul {{ margin: 0 0 11px 22px; padding: 0; }}
    li {{ margin: 3px 0; }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      background: #f3f4f6;
      color: #7f1d1d;
      padding: 1px 4px;
      border-radius: 3px;
      font-size: 0.92em;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 10px 12px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      margin: 8px 0 12px;
      break-inside: avoid;
    }}
    pre code {{
      background: transparent;
      color: #111827;
      padding: 0;
      border-radius: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 9px 0 13px;
      table-layout: fixed;
      break-inside: avoid;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 6px 7px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{ background: #eef2f7; text-align: left; }}
    hr {{ border: 0; border-top: 1px solid #e5e7eb; margin: 16px 0; }}
  </style>
</head>
<body>
{chr(10).join(body)}
</body>
</html>
"""


def x(text: str) -> str:
    return xml_escape(text, {'"': "&quot;", "'": "&apos;"})


def w_text(text: str) -> str:
    if text == "":
        return "<w:t/>"
    preserve = text[0].isspace() or text[-1].isspace() or "  " in text or "\t" in text
    attr = ' xml:space="preserve"' if preserve else ""
    return f"<w:t{attr}>{x(text)}</w:t>"


def run(text: str, *, code: bool = False, bold: bool = False) -> str:
    props: list[str] = []
    if code:
        props.append('<w:rStyle w:val="CodeChar"/>')
    if bold:
        props.append("<w:b/>")
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f"<w:r>{rpr}{w_text(text)}</w:r>"


def inline_runs(text: str, *, bold: bool = False) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append(run(part[1:-1], code=True, bold=bold))
        else:
            out.append(run(part, bold=bold))
    return "".join(out)


def paragraph(runs: str, *, style: str | None = None, before: int = 0, after: int = 120, left: int = 0, hanging: int = 0) -> str:
    props: list[str] = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    props.append(f'<w:spacing w:before="{before}" w:after="{after}" w:line="276" w:lineRule="auto"/>')
    if left or hanging:
        hanging_attr = f' w:hanging="{hanging}"' if hanging else ""
        props.append(f'<w:ind w:left="{left}"{hanging_attr}/>')
    return f"<w:p><w:pPr>{''.join(props)}</w:pPr>{runs}</w:p>"


def table_xml(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    cols = max(len(row) for row in rows)
    col_w = max(1200, int(9360 / cols))
    grid = "".join(f'<w:gridCol w:w="{col_w}"/>' for _ in range(cols))
    out = [
        "<w:tbl>",
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders><w:top w:val=\"single\" w:sz=\"4\" w:color=\"D1D5DB\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:color=\"D1D5DB\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"D1D5DB\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:color=\"D1D5DB\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"D1D5DB\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"D1D5DB\"/></w:tblBorders></w:tblPr>",
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]
    for ridx, row in enumerate(rows):
        cells = row + [""] * (cols - len(row))
        out.append("<w:tr>")
        for cell in cells:
            shade = '<w:shd w:fill="EEF2F7"/>' if ridx == 0 else ""
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{col_w}" w:type="dxa"/>{shade}</w:tcPr>')
            out.append(paragraph(inline_runs(cell, bold=(ridx == 0)), after=80))
            out.append("</w:tc>")
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def build_document_xml(blocks: list[Block]) -> str:
    body: list[str] = []
    for block in blocks:
        if block.kind == "heading":
            if block.level == 1:
                body.append(paragraph(inline_runs(block.text, bold=True), style="Title", after=260))
            else:
                style = f"Heading{min(block.level - 1, 3)}"
                body.append(paragraph(inline_runs(block.text, bold=True), style=style, before=220, after=120))
        elif block.kind == "paragraph":
            body.append(paragraph(inline_runs(block.text), after=140))
        elif block.kind == "list":
            for item in block.items or []:
                body.append(paragraph(run("- ") + inline_runs(item), after=70, left=360, hanging=240))
        elif block.kind == "code":
            lines = block.text.splitlines() or [""]
            for line in lines:
                body.append(paragraph(run(line if line else " ", code=True), style="CodeBlock", after=0))
        elif block.kind == "table":
            body.append(table_xml(block.rows or []))
        elif block.kind == "hr":
            body.append(paragraph(run(""), after=120))

    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}{sect}</w:body></w:document>"
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="23"/><w:szCs w:val="23"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="52"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:color w:val="0F172A"/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:color w:val="0F172A"/><w:sz w:val="29"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:color w:val="0F172A"/><w:sz w:val="25"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="CodeBlock"><w:name w:val="Code Block"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:sz w:val="20"/></w:rPr></w:style>
  <w:style w:type="character" w:styleId="CodeChar"><w:name w:val="Code Char"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:color w:val="7F1D1D"/><w:sz w:val="21"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="D1D5DB"/><w:left w:val="single" w:sz="4" w:color="D1D5DB"/><w:bottom w:val="single" w:sz="4" w:color="D1D5DB"/><w:right w:val="single" w:sz="4" w:color="D1D5DB"/><w:insideH w:val="single" w:sz="4" w:color="D1D5DB"/><w:insideV w:val="single" w:sz="4" w:color="D1D5DB"/></w:tblBorders></w:tblPr></w:style>
</w:styles>
"""


def write_docx(blocks: list[Block], docx_out: Path = DOCX_OUT, title: str = "Dokumentim Profesional i Projektit") -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""
    settings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:zoom w:percent="100"/></w:settings>
"""
    font_table = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:fonts xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:font w:name="Arial"/><w:font w:name="Consolas"/></w:fonts>
"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{x(title)}</dc:title>
  <dc:creator>AIN_PROJECT</dc:creator>
  <cp:lastModifiedBy>AIN_PROJECT</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Word</Application></Properties>
"""
    parts = {
        "[Content_Types].xml": content_types,
        "_rels/.rels": rels,
        "word/_rels/document.xml.rels": doc_rels,
        "word/document.xml": build_document_xml(blocks),
        "word/styles.xml": styles_xml(),
        "word/settings.xml": settings,
        "word/fontTable.xml": font_table,
        "docProps/core.xml": core,
        "docProps/app.xml": app,
    }
    if docx_out.exists():
        docx_out.unlink()
    with zipfile.ZipFile(docx_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, value in parts.items():
            zf.writestr(name, value.encode("utf-8"))


def main() -> None:
    for doc in DOCUMENTS:
        source = doc["source"]
        html_out = doc["html"]
        docx_out = doc["docx"]
        title = doc["title"]
        if not source.exists():
            print(f"Skipped missing source: {source}")
            continue
        markdown = source.read_text(encoding="utf-8")
        blocks = parse_markdown(markdown)
        html_out.write_text(build_html(blocks, title=title), encoding="utf-8")
        write_docx(blocks, docx_out=docx_out, title=title)
        print(f"Wrote {html_out}")
        print(f"Wrote {docx_out}")


if __name__ == "__main__":
    main()
