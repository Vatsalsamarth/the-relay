"""Download the three papers at the exact versions and extract text per page -> data/pages.json"""
import json, os, io, httpx, pypdf
PAPERS = {"attention": ("1706.03762", 7, "Attention Is All You Need"),
          "moshi": ("2410.00037", 1, "Moshi"),
          "dchunk": ("2507.07955", 2, "Dynamic Chunking (H-Net)")}
os.makedirs("data", exist_ok=True)
pages = []
for key, (aid, v, title) in PAPERS.items():
    url = f"https://arxiv.org/pdf/{aid}v{v}"
    r = httpx.get(url, follow_redirects=True, timeout=120); r.raise_for_status()
    open(f"data/{key}_v{v}.pdf", "wb").write(r.content)
    for i, p in enumerate(pypdf.PdfReader(io.BytesIO(r.content)).pages, 1):
        pages.append({"paper": key, "title": title, "page": i, "text": p.extract_text() or ""})
    print(key, "ok")
json.dump(pages, open("data/pages.json", "w"))
