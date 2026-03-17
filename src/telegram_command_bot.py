from __future__ import annotations

import logging
import secrets
import signal
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from src import config
from src.firebase.client import add_document, get_document, get_documents
from src.firebase.trade_logger import get_daily_pnl
from src.runtime_overrides import get_all_heartbeat_overrides

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.telegram.commands")

API_BASE = "https://api.telegram.org"

BOTS = {
    "range": {
        "exchange": "binance",
        "label": "Trail Range",
        "heartbeat_default": config.HEARTBEAT_INTERVAL_SECONDS,
    },
    "crashbot": {
        "exchange": "binance-crashbot",
        "label": "CrashBot",
        "heartbeat_default": config.CRASHBOT_HEARTBEAT_TELEGRAM_SECONDS,
    },
    "infinity": {
        "exchange": "revolut-infinity",
        "label": "Infinity",
        "heartbeat_default": config.INF_HEARTBEAT_SECONDS,
    },
    "london": {
        "exchange": "revolut-london",
        "label": "London Breakout",
        "heartbeat_default": config.LON_HEARTBEAT_SECONDS,
    },
    "dca": {
        "exchange": "revolut-dca",
        "label": "DCA RSI",
        "heartbeat_default": config.DCA_HEARTBEAT_SECONDS,
    },
}

ALIAS = {
    "trail": "range",
    "trailrange": "range",
    "range": "range",
    "crash": "crashbot",
    "crashbot": "crashbot",
    "inf": "infinity",
    "infinity": "infinity",
    "lon": "london",
    "london": "london",
    "dca": "dca",
    "dcasri": "dca",
    "dcarsi": "dca",
}


class TelegramCommandBot:
    def __init__(self) -> None:
        self._token = config.TELEGRAM_BOT_TOKEN
        self._allowed_chat = str(config.TELEGRAM_CHAT_ID or "").strip()
        self._poll_seconds = max(1, config.TELEGRAM_COMMANDS_POLL_SECONDS)
        self._offset = 0
        self._running = True
        self._client = httpx.Client(timeout=30.0)
        self._pending_confirms: dict[str, dict[str, object]] = {}

        if not self._token or not self._allowed_chat:
            raise RuntimeError("TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant")

    def run(self) -> None:
        logger.info("🤖 Telegram Command Bot démarré")
        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._offset = max(self._offset, update.get("update_id", 0) + 1)
                    self._handle_update(update)
            except Exception as e:
                logger.warning("Polling Telegram erreur: %s", e)
                time.sleep(self._poll_seconds)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._client.close()

    def _api(self, method: str, payload: Optional[dict] = None) -> dict:
        url = f"{API_BASE}/bot{self._token}/{method}"
        resp = self._client.post(url, json=payload or {})
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API {method} failed: {data}")
        return data

    def _get_updates(self) -> list[dict]:
        payload = {
            "offset": self._offset,
            "timeout": 20,
            "allowed_updates": ["message"],
        }
        data = self._api("getUpdates", payload)
        return data.get("result", [])

    def _send(self, chat_id: str, text: str) -> None:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            self._api("sendMessage", payload)
        except Exception as e:
            logger.warning("Envoi Telegram échoué: %s", e)

    def _handle_update(self, update: dict) -> None:
        msg = update.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = str((msg.get("chat") or {}).get("id", ""))

        if not text or not chat_id:
            return

        if chat_id != self._allowed_chat:
            logger.warning("Message ignoré chat_id non autorisé: %s", chat_id)
            return

        response = self._dispatch(text)
        if response:
            self._send(chat_id, response)

    def _dispatch(self, text: str) -> str:
        raw = text.strip()
        if raw.startswith("/"):
            raw = raw[1:]
        parts = [p for p in raw.split() if p]
        if not parts:
            return self._help([])

        cmd = parts[0].split("@")[0].lower()
        args = parts[1:]

        try:
            if cmd in ("help", "start"):
                return self._help(args)
            if cmd == "pnl":
                return self._cmd_pnl(args)
            if cmd == "perf":
                return self._cmd_perf(args)
            if cmd == "alloc":
                return self._cmd_alloc(args)
            if cmd == "health":
                return self._cmd_health(args)
            if cmd == "hb":
                return self._cmd_hb(args)
            if cmd == "open":
                return self._cmd_open(args)
            if cmd == "last":
                return self._cmd_last(args)
            if cmd == "cmdlog":
                return self._cmd_cmdlog(args)
            if cmd == "logs":
                return self._cmd_logs(args)
            if cmd == "close":
                return self._cmd_close(args)
            if cmd == "confirm":
                return self._cmd_confirm(args)
            if cmd == "sl":
                return self._cmd_sl(args)
            if cmd == "tp":
                return self._cmd_tp(args)
            return "Commande inconnue. Utilise /help"
        except Exception as e:
            logger.exception("Commande échouée")
            return f"Erreur commande: {e}"

    def _help(self, args: list[str]) -> str:
        topic = (args[0].lower().strip() if args else "")
        if topic in ("general", "general", "all", "général", "overview"):
            topic = ""

        if topic in ("pnl",):
            return (
                "Aide /pnl\n"
                "- Usage: /pnl <bot|all> [today|7d|30d|90d]\n"
                "- Affiche le PnL net et le nombre de trades\n"
                "- Exemples:\n"
                "  /pnl all today\n"
                "  /pnl crashbot 7d"
            )
        if topic in ("perf",):
            return (
                "Aide /perf\n"
                "- Usage: /perf <bot|all> [7d|30d|90d]\n"
                "- Affiche pnl, trades, winrate, profit factor\n"
                "- Exemples:\n"
                "  /perf all 30d\n"
                "  /perf london 90d"
            )
        if topic in ("alloc", "allocation"):
            return (
                "Aide /alloc\n"
                "- Usage: /alloc get\n"
                "- Usage: /alloc set <crashbot|range|infinity|london> <pct_decimal>\n"
                "- pct_decimal entre 0 et 1\n"
                "- Exemples:\n"
                "  /alloc get\n"
                "  /alloc set crashbot 0.70\n"
                "  /alloc set infinity 0.80"
            )
        if topic in ("health",):
            return (
                "Aide /health\n"
                "- Usage: /health <bot|all>\n"
                "- Affiche âge heartbeat, positions OPEN, close_fail_24h\n"
                "- Exemples:\n"
                "  /health all\n"
                "  /health range"
            )
        if topic in ("hb", "heartbeat"):
            return (
                "Aide /hb\n"
                "- Usage: /hb get [bot|all]\n"
                "- Usage: /hb set <bot> <5m|10m|30m|300|600|1800>\n"
                "- Usage: /hb reset <bot|all>\n"
                "- Exemples:\n"
                "  /hb get all\n"
                "  /hb set crashbot 5m\n"
                "  /hb reset london"
            )
        if topic in ("open",):
            return (
                "Aide /open\n"
                "- Usage: /open <bot|all>\n"
                "- Affiche le nombre de positions ouvertes + symboles\n"
                "- Exemples:\n"
                "  /open all\n"
                "  /open crashbot"
            )
        if topic in ("close",):
            return (
                "Aide /close\n"
                "- Usage: /close now <bot> <symbol|all>\n"
                "- Enfile une fermeture manuelle d'urgence\n"
                "- Si target=all: confirmation obligatoire via /confirm <token>\n"
                "- Exemples:\n"
                "  /close now crashbot BTCUSDC\n"
                "  /close now london all"
            )
        if topic in ("confirm",):
            return (
                "Aide /confirm\n"
                "- Usage: /confirm <token>\n"
                "- Valide une action critique (ex: close all)\n"
                "- Validité token: 120 secondes\n"
                "- Exemple: /confirm 7K4Q9P"
            )
        if topic in ("sl",):
            return (
                "Aide /sl\n"
                "- Usage: /sl set <bot> <symbol> <price>\n"
                "- Bots supportés: range, crashbot, london\n"
                "- Exemples:\n"
                "  /sl set crashbot BTCUSDC 0.0810\n"
                "  /sl set london ETH-USD 3200"
            )
        if topic in ("tp",):
            return (
                "Aide /tp\n"
                "- Usage: /tp set <bot> <symbol> <price>\n"
                "- Bots supportés: range, crashbot, london\n"
                "- Exemples:\n"
                "  /tp set crashbot BTCUSDC 71000\n"
                "  /tp set london ETH-USD 3800"
            )
        if topic in ("last",):
            return (
                "Aide /last\n"
                "- Usage: /last <bot|all> [n]\n"
                "- n: 1..10 (défaut 3)\n"
                "- Affiche les derniers trades clôturés\n"
                "- Exemples:\n"
                "  /last all 5\n"
                "  /last infinity 3"
            )
        if topic in ("logs",):
            return (
                "Aide /logs\n"
                "- Usage: /logs <bot|all> [n]\n"
                "- Défaut: exclut HEARTBEAT pour limiter le bruit\n"
                "- n: 1..30 (défaut 10)\n"
                "- Exemples:\n"
                "  /logs crashbot 8\n"
                "  /logs all 5"
            )
        if topic in ("cmdlog",):
            return (
                "Aide /cmdlog\n"
                "- Usage: /cmdlog [n]\n"
                "- Affiche les derniers OPERATOR_COMMAND (audit)\n"
                "- n: 1..50 (défaut 10)\n"
                "- Exemple: /cmdlog 20"
            )

        return (
            "TradeX Commands\n"
            "- /pnl <bot|all> [today|7d|30d|90d]\n"
            "- /perf <bot|all> [7d|30d|90d]\n"
            "- /alloc get\n"
            "- /alloc set <crashbot|range|infinity|london> <pct_decimal>\n"
            "- /health <bot|all>\n"
            "- /hb get [bot|all]\n"
            "- /hb set <bot> <5m|10m|30m|300|600|1800>\n"
            "- /hb reset <bot|all>\n"
            "- /open <bot|all>\n"
            "- /last <bot|all> [n]\n"
            "- /cmdlog [n]\n"
            "- /logs <bot|all> [n]\n"
            "- /close now <bot> <symbol|all>\n"
            "- /confirm <token>\n"
            "- /sl set <bot> <symbol> <price>\n"
            "- /tp set <bot> <symbol> <price>\n"
            "Tips: /help alloc | /help hb | /help logs | /help cmdlog | /help close | /help confirm\n"
            "Exemple: /pnl crashbot today"
        )

    def _parse_ts(self, value: object) -> Optional[datetime]:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str) and value:
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                return None
        else:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _fmt_ts(self, value: object) -> str:
        dt = self._parse_ts(value)
        return dt.strftime("%m-%d %H:%M") if dt else "n/a"

    def _resolve_bot(self, raw: str) -> str:
        key = raw.lower().strip()
        if key in ALIAS:
            return ALIAS[key]
        if key in BOTS:
            return key
        raise ValueError(f"bot inconnu: {raw}")

    def _parse_window(self, raw: str) -> tuple[str, datetime]:
        token = raw.lower().strip()
        now = datetime.now(timezone.utc)
        if token == "today":
            start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            return "today", start
        if token.endswith("d") and token[:-1].isdigit():
            days = int(token[:-1])
            if days <= 0:
                raise ValueError("fenêtre invalide")
            return token, now - timedelta(days=days)
        raise ValueError("fenêtre invalide (today|7d|30d|90d)")

    def _closed_trades(self, exchange: str, since: Optional[datetime]) -> list[dict]:
        filters = [("exchange", "==", exchange), ("status", "==", "CLOSED")]
        if since is not None:
            filters.append(("closed_at", ">=", since.isoformat()))
        return get_documents("trades", filters=filters)

    def _metrics(self, trades: list[dict]) -> tuple[float, int, float, float]:
        pnl = 0.0
        wins = 0
        losses = 0
        gross_win = 0.0
        gross_loss = 0.0
        for trade in trades:
            value = trade.get("pnl_net_usd")
            if value is None:
                value = trade.get("pnl_usd", 0.0)
            try:
                p = float(value)
            except Exception:
                p = 0.0
            pnl += p
            if p > 0:
                wins += 1
                gross_win += p
            else:
                losses += 1
                gross_loss += abs(p)

        n = len(trades)
        wr = (wins / n * 100.0) if n else 0.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
        return pnl, n, wr, pf

    def _cmd_pnl(self, args: list[str]) -> str:
        target = (args[0] if args else "all").lower()
        window = args[1] if len(args) > 1 else "today"
        label, start = self._parse_window(window)

        bot_keys = list(BOTS.keys()) if target == "all" else [self._resolve_bot(target)]
        lines = [f"PnL {label} UTC"]
        total = 0.0
        total_n = 0

        for bot in bot_keys:
            exchange = BOTS[bot]["exchange"]
            bot_label = BOTS[bot]["label"]
            if label == "today":
                pnl, n = get_daily_pnl(exchange)
            else:
                trades = self._closed_trades(exchange, start)
                pnl, n, _, _ = self._metrics(trades)
            total += pnl
            total_n += n
            lines.append(f"- {bot_label}: {pnl:+.2f}$ ({n} trades)")

        if len(bot_keys) > 1:
            lines.append(f"- Total: {total:+.2f}$ ({total_n} trades)")
        return "\n".join(lines)

    def _cmd_perf(self, args: list[str]) -> str:
        target = (args[0] if args else "all").lower()
        window = args[1] if len(args) > 1 else "30d"
        label, start = self._parse_window(window)

        bot_keys = list(BOTS.keys()) if target == "all" else [self._resolve_bot(target)]
        lines = [f"Perf {label} UTC"]

        for bot in bot_keys:
            exchange = BOTS[bot]["exchange"]
            bot_label = BOTS[bot]["label"]
            trades = self._closed_trades(exchange, start)
            pnl, n, wr, pf = self._metrics(trades)
            lines.append(
                f"- {bot_label}: pnl {pnl:+.2f}$ | trades {n} | WR {wr:.1f}% | PF {pf:.2f}"
            )

        return "\n".join(lines)

    def _cmd_alloc(self, args: list[str]) -> str:
        if not args or args[0].lower() == "get":
            current = get_document("allocation", "current") or {}
            runtime = get_document("runtime_overrides", "allocation") or {}

            lines = ["Allocation"]
            if current:
                lines.append(
                    "- Binance dyn: CrashBot "
                    f"{current.get('crash_pct', 0)*100:.0f}% | Range {current.get('trail_pct', 0)*100:.0f}%"
                )
            else:
                lines.append("- Binance dyn: indisponible")

            rev = runtime.get("revolut", {})
            if rev:
                lines.append(
                    f"- Revolut override: Infinity {rev.get('infinity_pct', 0)*100:.0f}% | "
                    f"London {rev.get('london_pct', 0)*100:.0f}%"
                )
            else:
                lines.append(
                    f"- Revolut défaut: Infinity {config.INF_CAPITAL_PCT*100:.0f}% | London {config.LON_CAPITAL_PCT*100:.0f}%"
                )
            lines.append("Note: overrides Firestore = intention opérateur; application bot selon implémentation runtime.")
            return "\n".join(lines)

        if len(args) != 3 or args[0].lower() != "set":
            return "Usage: /alloc set <crashbot|range|infinity|london> <pct_decimal>"

        bot = self._resolve_bot(args[1])
        pct = float(args[2])
        if pct < 0 or pct > 1:
            return "pct doit être entre 0 et 1"

        doc = get_document("runtime_overrides", "allocation") or {}
        now = datetime.now(timezone.utc).isoformat()

        if bot in ("infinity", "london"):
            inf = pct if bot == "infinity" else 1 - pct
            lon = pct if bot == "london" else 1 - pct
            doc["revolut"] = {
                "infinity_pct": round(inf, 4),
                "london_pct": round(lon, 4),
                "updated_at": now,
            }
        else:
            crash = pct if bot == "crashbot" else 1 - pct
            trail = pct if bot == "range" else 1 - pct
            doc["binance"] = {
                "crash_pct": round(crash, 4),
                "trail_pct": round(trail, 4),
                "updated_at": now,
            }

        add_document("runtime_overrides", doc, doc_id="allocation")
        add_document("events", {
            "event_type": "OPERATOR_COMMAND",
            "symbol": None,
            "exchange": BOTS[bot]["exchange"],
            "timestamp": now,
            "data": {
                "command": "alloc_set",
                "bot": bot,
                "pct": pct,
            },
        })
        return "Allocation override enregistrée."

    def _cmd_health(self, args: list[str]) -> str:
        target = (args[0] if args else "all").lower()
        bot_keys = list(BOTS.keys()) if target == "all" else [self._resolve_bot(target)]
        hb_overrides = get_all_heartbeat_overrides()

        lines = ["Santé bots"]
        now = datetime.now(timezone.utc)
        cutoff_dt = now - timedelta(hours=24)

        for bot in bot_keys:
            meta = BOTS[bot]
            exchange = meta["exchange"]
            hb = get_documents(
                "events",
                filters=[("event_type", "==", "HEARTBEAT"), ("exchange", "==", exchange)],
                order_by="timestamp",
                limit=1,
            )
            open_trades = get_documents(
                "trades",
                filters=[("exchange", "==", exchange), ("status", "==", "OPEN")],
            )
            failure_events = get_documents(
                "events",
                filters=[
                    ("event_type", "==", "CLOSE_FAILURE"),
                    ("exchange", "==", exchange),
                ],
            )
            failures_24h = 0
            for ev in failure_events:
                ts = ev.get("timestamp")
                try:
                    if isinstance(ts, datetime):
                        dt = ts
                    elif isinstance(ts, str) and ts:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = None
                    if dt is None:
                        continue
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt >= cutoff_dt:
                        failures_24h += 1
                except Exception:
                    continue

            hb_age_min = 9999.0
            if hb:
                ts = hb[0].get("timestamp")
                try:
                    if isinstance(ts, datetime):
                        dt = ts
                    elif isinstance(ts, str) and ts:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = None
                    if dt is not None:
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        hb_age_min = (now - dt).total_seconds() / 60.0
                except Exception:
                    pass

            hb_expected_raw = hb_overrides.get(bot, meta["heartbeat_default"])
            try:
                if hb_expected_raw is None:
                    hb_expected_raw = meta["heartbeat_default"]
                hb_expected = max(1, int(hb_expected_raw))
            except Exception:
                hb_expected = max(1, int(meta["heartbeat_default"]))
            ok = hb_age_min <= (hb_expected / 60.0) * 2.5
            emoji = "🟢" if ok else "🔴"
            lines.append(
                f"- {emoji} {meta['label']}: hb {hb_age_min:.0f}m (target {hb_expected//60}m) | "
                f"open {len(open_trades)} | close_fail_24h {failures_24h}"
            )

        return "\n".join(lines)

    def _parse_hb_seconds(self, raw: str) -> int:
        token = raw.lower().strip()
        mapping = {"5m": 300, "10m": 600, "30m": 1800}
        if token in mapping:
            return mapping[token]
        if token.isdigit():
            sec = int(token)
            if sec in (300, 600, 1800):
                return sec
        raise ValueError("Valeur heartbeat invalide (5m|10m|30m|300|600|1800)")

    def _cmd_hb(self, args: list[str]) -> str:
        if not args:
            return "Usage: /hb <get|set|reset> ..."

        action = args[0].lower()
        now = datetime.now(timezone.utc).isoformat()
        doc = get_document("runtime_overrides", "heartbeat") or {}
        values = doc.get("values", {})
        if not isinstance(values, dict):
            values = {}

        if action == "get":
            if len(args) == 2 and args[1].lower() != "all":
                bot = self._resolve_bot(args[1])
                value = values.get(bot, BOTS[bot]["heartbeat_default"])
                return f"Heartbeat {BOTS[bot]['label']}: {int(value)}s"

            lines = ["Heartbeat runtime"]
            for bot in BOTS:
                value = values.get(bot, BOTS[bot]["heartbeat_default"])
                source = "override" if bot in values else "default"
                lines.append(f"- {BOTS[bot]['label']}: {int(value)}s ({source})")
            return "\n".join(lines)

        if action == "set":
            if len(args) != 3:
                return "Usage: /hb set <bot> <5m|10m|30m|300|600|1800>"

            bot = self._resolve_bot(args[1])
            seconds = self._parse_hb_seconds(args[2])
            values[bot] = seconds
            doc["values"] = values
            doc["updated_at"] = now

            add_document("runtime_overrides", doc, doc_id="heartbeat")
            add_document("events", {
                "event_type": "OPERATOR_COMMAND",
                "symbol": None,
                "exchange": BOTS[bot]["exchange"],
                "timestamp": now,
                "data": {
                    "command": "hb_set",
                    "bot": bot,
                    "seconds": seconds,
                },
            })
            return f"Heartbeat {BOTS[bot]['label']} override enregistré à {seconds}s (live)."

        if action == "reset":
            if len(args) != 2:
                return "Usage: /hb reset <bot|all>"

            target = args[1].lower()
            if target == "all":
                values = {}
                doc["values"] = values
                doc["updated_at"] = now
                add_document("runtime_overrides", doc, doc_id="heartbeat")
                add_document("events", {
                    "event_type": "OPERATOR_COMMAND",
                    "symbol": None,
                    "exchange": "all",
                    "timestamp": now,
                    "data": {
                        "command": "hb_reset",
                        "bot": "all",
                        "seconds": None,
                    },
                })
                return "Heartbeats reset: tous les bots reviennent au défaut config."

            bot = self._resolve_bot(target)
            if bot in values:
                del values[bot]
            doc["values"] = values
            doc["updated_at"] = now
            add_document("runtime_overrides", doc, doc_id="heartbeat")
            add_document("events", {
                "event_type": "OPERATOR_COMMAND",
                "symbol": None,
                "exchange": BOTS[bot]["exchange"],
                "timestamp": now,
                "data": {
                    "command": "hb_reset",
                    "bot": bot,
                    "seconds": None,
                },
            })
            return f"Heartbeat reset pour {BOTS[bot]['label']} (retour config)."

        return "Usage: /hb <get|set|reset> ..."

    def _cmd_open(self, args: list[str]) -> str:
        target = (args[0] if args else "all").lower()
        bot_keys = list(BOTS.keys()) if target == "all" else [self._resolve_bot(target)]

        lines = ["Positions ouvertes"]
        total = 0
        for bot in bot_keys:
            meta = BOTS[bot]
            open_trades = get_documents(
                "trades",
                filters=[("exchange", "==", meta["exchange"]), ("status", "==", "OPEN")],
            )
            total += len(open_trades)
            symbols = sorted({str(t.get("symbol", "?")) for t in open_trades})
            symbols_str = ", ".join(symbols[:6]) if symbols else "—"
            if len(symbols) > 6:
                symbols_str += ", ..."
            lines.append(f"- {meta['label']}: {len(open_trades)} | {symbols_str}")

        if len(bot_keys) > 1:
            lines.append(f"- Total: {total}")
        return "\n".join(lines)

    def _cmd_last(self, args: list[str]) -> str:
        target = (args[0] if args else "all").lower()
        bot_keys = list(BOTS.keys()) if target == "all" else [self._resolve_bot(target)]

        n = 3
        if len(args) > 1:
            try:
                n = int(args[1])
            except Exception:
                return "Usage: /last <bot|all> [n]"
        n = max(1, min(n, 10))

        lines = [f"Derniers trades clos (n={n})"]
        for bot in bot_keys:
            meta = BOTS[bot]
            rows = get_documents(
                "trades",
                filters=[("exchange", "==", meta["exchange"]), ("status", "==", "CLOSED")],
                order_by="closed_at",
                limit=n,
            )
            if not rows:
                lines.append(f"- {meta['label']}: aucun trade clos")
                continue

            lines.append(f"- {meta['label']}")
            for row in rows:
                pnl_raw = row.get("pnl_net_usd")
                if pnl_raw is None:
                    pnl_raw = row.get("pnl_usd", 0.0)
                try:
                    pnl = float(pnl_raw)
                except Exception:
                    pnl = 0.0

                ts = row.get("closed_at") or row.get("timestamp")
                ts_str = self._fmt_ts(ts)

                symbol = row.get("symbol", "?")
                lines.append(f"  • {ts_str} | {symbol} | {pnl:+.2f}$")

        return "\n".join(lines)

    def _cmd_cmdlog(self, args: list[str]) -> str:
        n = 10
        if args:
            try:
                n = int(args[0])
            except Exception:
                return "Usage: /cmdlog [n]"
        n = max(1, min(n, 50))

        rows = get_documents(
            "events",
            filters=[("event_type", "==", "OPERATOR_COMMAND")],
        )

        def _sort_key(row: dict) -> float:
            dt = self._parse_ts(row.get("timestamp"))
            return dt.timestamp() if dt else 0.0

        rows_sorted = sorted(rows, key=_sort_key, reverse=True)[:n]
        if not rows_sorted:
            return "Command log: aucun événement opérateur."

        lines = [f"Command log (last {len(rows_sorted)})"]
        for row in rows_sorted:
            raw_data = row.get("data")
            data: dict[str, object]
            if isinstance(raw_data, dict):
                data = raw_data
            else:
                data = {}
            cmd = str(data.get("command", "?"))
            bot = str(data.get("bot", "-"))
            exchange = str(row.get("exchange", "-"))

            value = "-"
            sec_raw = data.get("seconds")
            pct_raw = data.get("pct")
            if sec_raw is not None:
                value = f"{int(str(sec_raw))}s"
            elif pct_raw is not None:
                try:
                    value = f"{float(str(pct_raw)):.4f}"
                except Exception:
                    value = str(pct_raw)

            lines.append(
                f"- {self._fmt_ts(row.get('timestamp'))} | {cmd} | bot={bot} | value={value} | ex={exchange} | ok"
            )
        return "\n".join(lines)

    def _cmd_logs(self, args: list[str]) -> str:
        if not args:
            return "Usage: /logs <bot|all> [n]"

        target = args[0].lower()
        bot_keys = list(BOTS.keys()) if target == "all" else [self._resolve_bot(target)]

        n = 10
        if len(args) > 1:
            try:
                n = int(args[1])
            except Exception:
                return "Usage: /logs <bot|all> [n]"
        n = max(1, min(n, 30))

        lines = [f"Logs récents (n={n}, sans HEARTBEAT)"]
        for bot in bot_keys:
            meta = BOTS[bot]
            events = get_documents(
                "events",
                filters=[("exchange", "==", meta["exchange"])],
            )
            filtered = [e for e in events if str(e.get("event_type", "")) != "HEARTBEAT"]

            def _sort_key(row: dict) -> float:
                dt = self._parse_ts(row.get("timestamp"))
                return dt.timestamp() if dt else 0.0

            rows = sorted(filtered, key=_sort_key, reverse=True)[:n]
            lines.append(f"- {meta['label']}")
            if not rows:
                lines.append("  • aucun event récent")
                continue

            for row in rows:
                event_type = str(row.get("event_type", "?"))
                symbol = str(row.get("symbol") or "-")
                raw_data = row.get("data")
                data: dict[str, object]
                if isinstance(raw_data, dict):
                    data = raw_data
                else:
                    data = {}

                detail = ""
                if event_type == "CLOSE_FAILURE":
                    detail = f"attempt={data.get('attempt', '?')}"
                elif event_type == "TREND_CHANGE":
                    detail = f"{data.get('old', '?')}→{data.get('new', '?')}"
                elif event_type == "OPERATOR_COMMAND":
                    detail = str(data.get("command", "?"))

                suffix = f" | {detail}" if detail else ""
                lines.append(
                    f"  • {self._fmt_ts(row.get('timestamp'))} | {event_type} | {symbol}{suffix}"
                )

        return "\n".join(lines)

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.upper().strip()

    def _build_confirm_token(self) -> str:
        return secrets.token_hex(3).upper()

    def _queue_runtime_action(self, bot: str, action: str, symbol: str, value: Optional[float]) -> str:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "bot": bot,
            "action": action,
            "symbol": self._normalize_symbol(symbol),
            "value": value,
            "status": "pending",
            "requested_at": now,
            "requested_by_chat": self._allowed_chat,
            "source": "telegram",
        }
        action_id = add_document("runtime_actions", payload)
        add_document("events", {
            "event_type": "OPERATOR_COMMAND",
            "symbol": self._normalize_symbol(symbol),
            "exchange": BOTS[bot]["exchange"],
            "timestamp": now,
            "data": {
                "command": action,
                "bot": bot,
                "value": value,
                "action_id": action_id,
            },
        })
        return action_id or "n/a"

    def _cmd_close(self, args: list[str]) -> str:
        if len(args) != 3 or args[0].lower() != "now":
            return "Usage: /close now <bot> <symbol|all>"

        bot = self._resolve_bot(args[1])
        symbol = self._normalize_symbol(args[2])

        if symbol == "ALL":
            token = self._build_confirm_token()
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=120)
            self._pending_confirms[token] = {
                "action": "close",
                "bot": bot,
                "symbol": "ALL",
                "value": None,
                "expires_at": expires_at,
            }
            now = datetime.now(timezone.utc).isoformat()
            add_document("events", {
                "event_type": "OPERATOR_COMMAND",
                "symbol": "ALL",
                "exchange": BOTS[bot]["exchange"],
                "timestamp": now,
                "data": {
                    "command": "close_request",
                    "bot": bot,
                    "token": token,
                    "expires_at": expires_at.isoformat(),
                },
            })
            return (
                f"⚠️ Confirmation requise pour close ALL sur {bot}. "
                f"Confirme avec /confirm {token} (valide 120s)."
            )

        action_id = self._queue_runtime_action(bot, "close", symbol, None)
        return f"Close manuel enfilé: bot={bot}, symbol={symbol}, action_id={action_id}"

    def _cmd_confirm(self, args: list[str]) -> str:
        if len(args) != 1:
            return "Usage: /confirm <token>"

        token = args[0].upper().strip()
        item = self._pending_confirms.get(token)
        if not item:
            return "Token inconnu ou expiré."

        expires_at = item.get("expires_at")
        if not isinstance(expires_at, datetime) or datetime.now(timezone.utc) > expires_at:
            self._pending_confirms.pop(token, None)
            return "Token expiré."

        action = str(item.get("action", ""))
        bot = str(item.get("bot", ""))
        symbol = str(item.get("symbol", ""))
        value_raw = item.get("value")

        self._pending_confirms.pop(token, None)

        if action != "close":
            return "Action non confirmable."

        action_id = self._queue_runtime_action(bot, "close", symbol, None if value_raw is None else float(str(value_raw)))
        now = datetime.now(timezone.utc).isoformat()
        add_document("events", {
            "event_type": "OPERATOR_COMMAND",
            "symbol": symbol,
            "exchange": BOTS[bot]["exchange"],
            "timestamp": now,
            "data": {
                "command": "close_confirm",
                "bot": bot,
                "token": token,
                "action_id": action_id,
            },
        })
        return f"✅ Confirmation reçue. Close enfilé: bot={bot}, symbol={symbol}, action_id={action_id}"

    def _cmd_sl(self, args: list[str]) -> str:
        if len(args) != 4 or args[0].lower() != "set":
            return "Usage: /sl set <bot> <symbol> <price>"

        bot = self._resolve_bot(args[1])
        if bot == "infinity":
            return "SL manuel non supporté sur infinity (utiliser /close now infinity <symbol>)."

        symbol = self._normalize_symbol(args[2])
        try:
            price = float(args[3])
        except Exception:
            return "Prix invalide"
        if price <= 0:
            return "Prix invalide"

        action_id = self._queue_runtime_action(bot, "set_sl", symbol, price)
        return f"SL manuel enfilé: bot={bot}, symbol={symbol}, sl={price}, action_id={action_id}"

    def _cmd_tp(self, args: list[str]) -> str:
        if len(args) != 4 or args[0].lower() != "set":
            return "Usage: /tp set <bot> <symbol> <price>"

        bot = self._resolve_bot(args[1])
        if bot == "infinity":
            return "TP manuel non supporté sur infinity (utiliser /close now infinity <symbol>)."

        symbol = self._normalize_symbol(args[2])
        try:
            price = float(args[3])
        except Exception:
            return "Prix invalide"
        if price <= 0:
            return "Prix invalide"

        action_id = self._queue_runtime_action(bot, "set_tp", symbol, price)
        return f"TP manuel enfilé: bot={bot}, symbol={symbol}, tp={price}, action_id={action_id}"


def main() -> None:
    bot = TelegramCommandBot()

    def _stop(_sig, _frame):
        bot.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        bot.run()
    finally:
        bot.close()


if __name__ == "__main__":
    main()
