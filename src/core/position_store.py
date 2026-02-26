"""
Persistance des positions et ranges sur disque (JSON).

Permet de ne pas perdre l'état en cas de redémarrage du bot.
Fichier atomique : écriture dans un .tmp puis rename pour éviter la corruption.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from src.core.models import Position, RangeState

logger = logging.getLogger("tradex.store")

# Chemin par défaut : data/state.json à la racine du projet
_DEFAULT_STATE_FILE = os.environ.get(
    "TRADEX_STATE_FILE",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "state.json"),
)


class PositionStore:
    """Sauvegarde et chargement atomique des positions + ranges."""

    def __init__(self, state_file: Optional[str] = None) -> None:
        self._path = Path(state_file or _DEFAULT_STATE_FILE).resolve()

    # ── Sauvegarde ─────────────────────────────────────────────────────────────

    def save(
        self,
        positions: dict[str, Position],
        ranges: dict[str, RangeState],
    ) -> None:
        """Sauvegarde l'état complet sur disque (atomique)."""
        state = {
            "positions": {
                sym: pos.to_dict() for sym, pos in positions.items()
            },
            "ranges": {
                sym: rs.to_dict() for sym, rs in ranges.items()
            },
        }

        # Créer le dossier si nécessaire
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Écriture atomique : .tmp → rename
        tmp_path = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            tmp_path.replace(self._path)
            logger.debug("💾 État sauvegardé (%d positions, %d ranges)",
                         len(positions), len(ranges))
        except Exception as e:
            logger.error("❌ Échec de la sauvegarde d'état: %s", e)
            # Nettoyer le fichier temporaire si possible
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    # ── Chargement ─────────────────────────────────────────────────────────────

    def load(self) -> tuple[dict[str, Position], dict[str, RangeState]]:
        """Charge l'état depuis le disque. Retourne des dicts vides si pas de fichier."""
        positions: dict[str, Position] = {}
        ranges: dict[str, RangeState] = {}

        if not self._path.exists():
            logger.info("📂 Pas de fichier d'état trouvé (%s) — démarrage à vide", self._path)
            return positions, ranges

        try:
            with open(self._path, "r") as f:
                state = json.load(f)

            # Charger les positions
            for sym, data in state.get("positions", {}).items():
                try:
                    positions[sym] = Position.from_dict(data)
                except Exception as e:
                    logger.warning("⚠️ Position %s corrompue, ignorée: %s", sym, e)

            # Charger les ranges
            for sym, data in state.get("ranges", {}).items():
                try:
                    ranges[sym] = RangeState.from_dict(data)
                except Exception as e:
                    logger.warning("⚠️ Range %s corrompu, ignoré: %s", sym, e)

            logger.info(
                "📂 État chargé: %d positions, %d ranges depuis %s",
                len(positions), len(ranges), self._path,
            )
        except json.JSONDecodeError as e:
            logger.error("❌ Fichier d'état corrompu (%s): %s — démarrage à vide", self._path, e)
        except Exception as e:
            logger.error("❌ Impossible de charger l'état: %s — démarrage à vide", e)

        return positions, ranges

    # ── Utilitaire ─────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Chemin du fichier d'état."""
        return self._path
