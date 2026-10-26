from fastapi import APIRouter

from app.api.v1.endpoints import classify, moderation_status, reports, review_queue

api_router = APIRouter()
api_router.include_router(classify.router, tags=["clasificación"])
api_router.include_router(reports.router, tags=["reportes"])
api_router.include_router(review_queue.router, tags=["revisión humana"])
api_router.include_router(moderation_status.router, tags=["estado"])
