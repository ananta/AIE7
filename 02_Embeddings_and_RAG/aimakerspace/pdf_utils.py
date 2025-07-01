import os
from typing import List, Tuple
from pypdf import PdfReader


class UnsupportedFileTypeError(Exception):
    pass


class PDFLoader:
    def __init__(self, path: str):
        self.documents = []
        self.path = path

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_file()
        else:
            raise UnsupportedFileTypeError(
                "Provided path is not a valid directory or a .pdf file."
            )

    def load_file(self):
        reader = PdfReader(self.path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += page.extract_text() or ""
        self.documents.append(
            (text, {"source": self.path, "page_count": len(reader.pages)})
        )

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        reader = PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() or ""
                        self.documents.append(
                            (
                                text,
                                {
                                    "source": file_path,
                                    "page_count": len(reader.pages),
                                },
                            )
                        )

    def load_documents(self) -> List[Tuple[str, dict]]:
        self.load()
        return self.documents 