from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class DocumentNotFoundError(Exception):
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        super().__init__(f"document not found: {doc_id}")


class EmbeddingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class VectorStoreError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _error_response(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "detail": detail},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(
        request: Request, exc: DocumentNotFoundError
    ) -> JSONResponse:
        return _error_response(404, "document_not_found", str(exc))

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(
        request: Request, exc: EmbeddingError
    ) -> JSONResponse:
        return _error_response(502, "embedding_error", str(exc))

    @app.exception_handler(VectorStoreError)
    async def vector_store_error_handler(
        request: Request, exc: VectorStoreError
    ) -> JSONResponse:
        return _error_response(500, "vector_store_error", str(exc))
