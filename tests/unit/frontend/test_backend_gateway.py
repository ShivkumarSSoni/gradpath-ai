"""Unit tests for the frontend gateway's own logic (not Streamlit widgets)."""

import io

import pandas as pd
import pytest

from gradpath.frontend import backend_gateway


class _FakeUpload:
    """Stands in for Streamlit's UploadedFile: has .size and .getvalue()."""

    def __init__(self, content: bytes):
        self._content = content
        self.size = len(content)
        self.name = "upload.csv"

    def getvalue(self) -> bytes:
        return self._content


def test_read_uploaded_file_returns_none_none_when_nothing_was_uploaded():
    records, error = backend_gateway.read_uploaded_file(None)

    assert records is None
    assert error is None


def test_read_uploaded_file_rejects_a_file_over_the_size_cap(monkeypatch):
    monkeypatch.setattr(backend_gateway, "MAX_UPLOAD_FILE_SIZE_BYTES", 10)
    oversized_upload = _FakeUpload(b"a" * 100)

    records, error = backend_gateway.read_uploaded_file(oversized_upload)

    assert records is None
    assert "limit" in error


def test_read_uploaded_file_rejects_content_that_is_not_a_valid_csv():
    # A header with 2 columns but a row with far more fields is a genuine
    # CSV parsing error (not just raggedness pandas tolerates).
    malformed_csv = _FakeUpload(b"a,b\n1,2,3,4,5,6,7,8,9,10\n")

    records, error = backend_gateway.read_uploaded_file(malformed_csv)

    assert records is None
    assert error is not None


def test_read_uploaded_file_parses_a_valid_csv():
    csv_bytes = b"Target\nDropout\nGraduate\n"

    records, error = backend_gateway.read_uploaded_file(_FakeUpload(csv_bytes))

    assert error is None
    assert list(records["Target"]) == ["Dropout", "Graduate"]
