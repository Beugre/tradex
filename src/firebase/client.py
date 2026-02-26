"""
Client Firestore pour TradeX.
Gère la connexion à Firebase et expose des helpers CRUD pour les collections.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.cloud import firestore
from google.oauth2 import service_account

from src import config

logger = logging.getLogger("tradex.firebase")

_db: Optional[firestore.Client] = None


def get_db() -> Optional[firestore.Client]:
    """Retourne l'instance Firestore (singleton). None si désactivé ou erreur."""
    global _db
    if not config.FIREBASE_ENABLED:
        return None
    if _db is not None:
        return _db
    try:
        cred_path = config.FIREBASE_CREDENTIALS_PATH
        if not cred_path.exists():
            logger.error("❌ Firebase credentials introuvable: %s", cred_path)
            return None
        credentials = service_account.Credentials.from_service_account_file(
            str(cred_path)
        )
        _db = firestore.Client(
            project=credentials.project_id,
            credentials=credentials,
        )
        logger.info("🔥 Firebase connecté (project=%s)", credentials.project_id)
        return _db
    except Exception as e:
        logger.error("❌ Firebase init échoué: %s", e)
        return None


def add_document(collection: str, data: dict[str, Any], doc_id: Optional[str] = None) -> Optional[str]:
    """Ajoute un document dans une collection. Retourne le doc ID ou None."""
    db = get_db()
    if db is None:
        return None
    try:
        ref = db.collection(collection)
        if doc_id:
            ref.document(doc_id).set(data)
            return doc_id
        else:
            _, doc_ref = ref.add(data)
            return doc_ref.id
    except Exception as e:
        logger.error("❌ Firebase write [%s] échoué: %s", collection, e)
        return None


def update_document(collection: str, doc_id: str, data: dict[str, Any]) -> bool:
    """Met à jour un document existant. Retourne True si OK."""
    db = get_db()
    if db is None:
        return False
    try:
        db.collection(collection).document(doc_id).update(data)
        return True
    except Exception as e:
        logger.error("❌ Firebase update [%s/%s] échoué: %s", collection, doc_id, e)
        return False


def delete_documents_batch(
    collection: str,
    filters: list[tuple[str, str, Any]],
    batch_size: int = 100,
) -> int:
    """Supprime par lots les documents matchant les filtres. Retourne le nombre supprimé."""
    db = get_db()
    if db is None:
        return 0
    try:
        query = db.collection(collection)
        for field, op, value in filters:
            query = query.where(field, op, value)
        query = query.limit(batch_size)

        total = 0
        while True:
            docs = list(query.stream())
            if not docs:
                break
            batch = db.batch()
            for doc in docs:
                batch.delete(doc.reference)
            batch.commit()
            total += len(docs)
            if len(docs) < batch_size:
                break
        return total
    except Exception as e:
        logger.error("❌ Firebase delete_batch [%s] échoué: %s", collection, e)
        return 0


def get_documents(
    collection: str,
    filters: Optional[list[tuple[str, str, Any]]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Lit des documents avec filtres optionnels."""
    db = get_db()
    if db is None:
        return []
    try:
        query = db.collection(collection)
        if filters:
            for field, op, value in filters:
                query = query.where(field, op, value)
        if order_by:
            query = query.order_by(order_by, direction=firestore.Query.DESCENDING)
        if limit:
            query = query.limit(limit)
        return [
            {**doc.to_dict(), "_id": doc.id}
            for doc in query.stream()
        ]
    except Exception as e:
        logger.error("❌ Firebase read [%s] échoué: %s", collection, e)
        return []
