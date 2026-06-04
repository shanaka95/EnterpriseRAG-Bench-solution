"""
Celery application configuration.
"""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "rag_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,  # Auto-delete results after 1 hour to prevent Redis bloat
    worker_prefetch_multiplier=1,
    worker_concurrency=1,  # GPU tasks should run one at a time
    task_routes={
        "build_cluster_tree": {"queue": "gpu"},
        "ingest_and_build": {"queue": "gpu"},
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.tasks"])
