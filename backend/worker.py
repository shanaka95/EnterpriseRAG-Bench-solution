"""
Celery worker entry point.
Run with: celery -A worker.celery_app worker --loglevel=info --queue=gpu
"""
from app.tasks.celery_app import celery_app

# Import tasks so they are registered
import app.tasks.clustering  # noqa: F401

__all__ = ["celery_app"]
