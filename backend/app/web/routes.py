"""Thin server-rendered routes for knowledge-base management pages."""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

WEB_DIRECTORY = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIRECTORY / "templates"))

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
def knowledge_base_list_page(request: Request) -> HTMLResponse:
    """Render the management landing page; data is loaded through the existing API."""
    return templates.TemplateResponse(request, "knowledge_bases.html")


@router.get("/knowledge-bases/{knowledge_base_id}", response_class=HTMLResponse)
def knowledge_base_detail_page(request: Request, knowledge_base_id: UUID) -> HTMLResponse:
    """Render one knowledge-base document management page."""
    return templates.TemplateResponse(
        request,
        "knowledge_base_detail.html",
        {"knowledge_base_id": str(knowledge_base_id)},
    )
