"""
Modèles de données partagés pour TradeX.
Structures typées utilisées dans tout le core (sans dépendance I/O).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


# ── Enums ──────────────────────────────────────────────────────────────────────

class TrendDirection(Enum):
    """Direction de la tendance selon la Dow Theory."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SwingType(Enum):
    """Classification d'un point de swing."""
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low


class SwingLevel(Enum):
    """Type brut d'un swing avant classification Dow."""
    HIGH = "HIGH"
    LOW = "LOW"


class OrderSide(Enum):
    """Sens de l'ordre."""
    BUY = "buy"
    SELL = "sell"


class PositionStatus(Enum):
    """État d'une position gérée par le bot."""
    PENDING = "PENDING"        # En attente de déclenchement du seuil
    OPEN = "OPEN"              # Position ouverte (ordre exécuté)
    ZERO_RISK = "ZERO_RISK"    # SL déplacé pour verrouiller du profit
    CLOSED = "CLOSED"          # Position fermée


class StrategyType(Enum):
    """Type de stratégie ayant ouvert la position."""
    TREND = "TREND"            # Trend following (Dow Theory)
    RANGE = "RANGE"            # Mean reversion (range)
    BREAKOUT = "BREAKOUT"      # Breakout Volatility Expansion


# ── Structures de données ──────────────────────────────────────────────────────

@dataclass
class Candle:
    """Bougie OHLCV."""
    timestamp: int      # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SwingPoint:
    """Point de swing détecté (sommet ou creux)."""
    index: int              # Index dans la liste de bougies
    price: float            # Prix du swing (high ou low de la bougie)
    level: SwingLevel       # HIGH ou LOW
    timestamp: int          # Timestamp de la bougie pivot
    swing_type: Optional[SwingType] = None  # HH/HL/LH/LL (classifié après)


@dataclass
class TrendState:
    """État courant de la tendance pour une paire."""
    symbol: str
    direction: TrendDirection = TrendDirection.NEUTRAL
    last_high: Optional[SwingPoint] = None      # Dernier sommet détecté
    last_low: Optional[SwingPoint] = None       # Dernier creux détecté
    prev_high: Optional[SwingPoint] = None      # Avant-dernier sommet
    prev_low: Optional[SwingPoint] = None       # Avant-dernier creux
    swings: list[SwingPoint] = field(default_factory=list)
    neutral_reason: Optional[str] = None        # Raison si NEUTRAL (pour les logs)

    @property
    def entry_level(self) -> Optional[float]:
        """Niveau de prix pour le déclenchement de l'entrée."""
        if self.direction == TrendDirection.BULLISH and self.last_high:
            return self.last_high.price
        elif self.direction == TrendDirection.BEARISH and self.last_low:
            return self.last_low.price
        return None

    @property
    def sl_level(self) -> Optional[float]:
        """Niveau de prix pour le stop loss."""
        if self.direction == TrendDirection.BULLISH and self.last_low:
            return self.last_low.price
        elif self.direction == TrendDirection.BEARISH and self.last_high:
            return self.last_high.price
        return None


@dataclass
class RangeState:
    """État d'un range détecté en régime NEUTRAL (mean-reversion)."""
    symbol: str
    range_high: float           # Borne haute du range
    range_low: float            # Borne basse du range
    is_valid: bool = True       # Range assez large et exploitable
    cooldown_until: int = 0     # Timestamp (Unix s) avant lequel on ne trade pas (post-cassure)

    @property
    def range_mid(self) -> float:
        """Milieu du range (cible TP pour mean-reversion)."""
        return (self.range_high + self.range_low) / 2

    @property
    def range_width_pct(self) -> float:
        """Largeur du range en pourcentage."""
        mid = self.range_mid
        if mid == 0:
            return 0.0
        return (self.range_high - self.range_low) / mid

    def to_dict(self) -> dict:
        """Sérialise le range en dictionnaire JSON-compatible."""
        return {
            "symbol": self.symbol,
            "range_high": self.range_high,
            "range_low": self.range_low,
            "is_valid": self.is_valid,
            "cooldown_until": self.cooldown_until,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RangeState":
        """Désérialise un dictionnaire en RangeState."""
        return cls(
            symbol=data["symbol"],
            range_high=data["range_high"],
            range_low=data["range_low"],
            is_valid=data.get("is_valid", True),
            cooldown_until=data.get("cooldown_until", 0),
        )


@dataclass
class OrderRequest:
    """Paramètres d'un ordre à placer sur Revolut X."""
    symbol: str
    side: OrderSide
    base_size: str          # Taille en unités de base (ex: "0.001" BTC)
    price: str              # Prix limit
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_api_payload(self) -> dict:
        """Convertit en payload JSON pour POST /orders."""
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_configuration": {
                "limit": {
                    "base_size": self.base_size,
                    "price": self.price,
                }
            },
        }


@dataclass
class Position:
    """Position gérée par le bot (pas un concept Revolut X natif)."""
    symbol: str
    side: OrderSide
    entry_price: float
    sl_price: float
    size: float                     # En unités de base
    venue_order_id: str             # ID de l'ordre d'entrée chez Revolut X
    status: PositionStatus = PositionStatus.PENDING
    strategy: StrategyType = StrategyType.TREND  # Qui a ouvert ce trade
    is_zero_risk_applied: bool = False
    zero_risk_sl: Optional[float] = None    # SL ajusté pour zero-risk/trailing
    peak_price: Optional[float] = None      # Plus haut (BUY) ou plus bas (SELL) atteint
    tp_price: Optional[float] = None        # Take Profit (utilisé par RANGE)
    pnl: Optional[float] = None             # P&L réalisé à la clôture
    firebase_trade_id: Optional[str] = None  # ID du doc Firebase pour tracking
    trailing_active: bool = False            # Trail@TP actif (après TP OCO rempli)
    trailing_steps: int = 0                  # Nombre de paliers franchis
    trailing_sl: Optional[float] = None      # SL trailing courant

    def to_dict(self) -> dict:
        """Sérialise la position en dictionnaire JSON-compatible."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "size": self.size,
            "venue_order_id": self.venue_order_id,
            "status": self.status.value,
            "strategy": self.strategy.value,
            "is_zero_risk_applied": self.is_zero_risk_applied,
            "zero_risk_sl": self.zero_risk_sl,
            "peak_price": self.peak_price,
            "tp_price": self.tp_price,
            "pnl": self.pnl,
            "firebase_trade_id": self.firebase_trade_id,
            "trailing_active": self.trailing_active,
            "trailing_steps": self.trailing_steps,
            "trailing_sl": self.trailing_sl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        """Désérialise un dictionnaire en Position."""
        return cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            entry_price=data["entry_price"],
            sl_price=data["sl_price"],
            size=data["size"],
            venue_order_id=data["venue_order_id"],
            status=PositionStatus(data["status"]),
            strategy=StrategyType(data.get("strategy", "TREND")),
            is_zero_risk_applied=data.get("is_zero_risk_applied", False),
            zero_risk_sl=data.get("zero_risk_sl"),
            peak_price=data.get("peak_price"),
            tp_price=data.get("tp_price"),
            pnl=data.get("pnl"),
            firebase_trade_id=data.get("firebase_trade_id"),
            trailing_active=data.get("trailing_active", False),
            trailing_steps=data.get("trailing_steps", 0),
            trailing_sl=data.get("trailing_sl"),
        )


@dataclass
class TickerData:
    """Données de ticker temps réel."""
    symbol: str
    bid: float
    ask: float
    mid: float
    last_price: float


@dataclass
class Balance:
    """Solde d'une devise sur le compte Revolut X."""
    currency: str
    available: float
    reserved: float
    total: float
