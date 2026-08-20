import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ─────────────────────────────────────────────────────────
# Step 1: Load raw text out of files
# ─────────────────────────────────────────────────────────

def load_txt(path: str) -> str:
    """Reads a plain text / markdown file and returns its contents as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    """Reads a PDF file using LangChain's PyPDFLoader and returns all its text."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(folder_path: str) -> list[dict]:
    """
    Reads every .txt / .md / .pdf file in a folder.

    Returns a list like:
        [{"source": "my_resume.pdf", "text": "...full text..."}, ...]
    """
    documents = []

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)
        extension = os.path.splitext(filename)[1].lower()

        if extension in (".txt", ".md"):
            text = load_txt(file_path)
        elif extension == ".pdf":
            text = load_pdf(file_path)
        else:
            continue  # skip file types we don't know how to read

        if text.strip():
            documents.append({"source": filename, "text": text})

    return documents


# ─────────────────────────────────────────────────────────
# Step 2: Split long text into small overlapping chunks using LangChain
# ─────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 300) -> list[str]:
    """
    Splits text recursively using LangChain's RecursiveCharacterTextSplitter.
    Preserves paragraph and sentence boundaries.
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_text(text)


# ─────────────────────────────────────────────────────────
# Step 3: Put it together — folder of files -> list of chunks ready to embed
# ─────────────────────────────────────────────────────────

def process_folder(folder_path: str) -> list[dict]:
    """
    Loads every document in a folder and splits each one into chunks.

    Returns a list like:
        [{"id": "my_resume.pdf-0", "text": "...", "source": "my_resume.pdf"}, ...]
    """
    documents = load_documents(folder_path)

    all_chunks = []
    for document in documents:
        text_chunks = chunk_text(document["text"])

        for index, chunk in enumerate(text_chunks):
            all_chunks.append({
                "id": f"{document['source']}-{index}",
                "text": chunk,
                "source": document["source"],
            })

    return all_chunks


if __name__ == "__main__":
    path = r"uploads\my_resume.pdf"

    chunks = []

    for chunk in chunk_text(text=load_pdf(path=path)):
        chunks.append(chunk)

    print(len(chunks))
    print(chunks[1])
