# 🧠 TradeX — Comment ça marche (version simple)

## Table des matières

1. [L'idée générale](#lidée-générale)
2. [Les trois stratégies](#les-trois-stratégies)
3. [Stratégie 1 — Trend Following (Dow Theory)](#stratégie-1--trend-following-dow-theory)
   - [Détecter les sommets et les creux](#détecter-les-sommets-et-les-creux)
   - [Identifier la tendance](#identifier-la-tendance)
   - [Décider quand entrer](#décider-quand-entrer-trend)
   - [Protéger ses gains (zero-risk + trailing)](#protéger-ses-gains-zero-risk--trailing)
   - [Sortir du trade](#sortir-du-trade-trend)
4. [Stratégie 2 — Mean Reversion Range](#stratégie-2--mean-reversion-range)
   - [Détecter le range](#détecter-le-range)
   - [Décider quand entrer](#décider-quand-entrer-range)
   - [Take Profit et Stop Loss](#take-profit-et-stop-loss-range)
   - [Cooldown après breakout](#cooldown-après-breakout)
5. [Stratégie 3 — Breakout Volatility Expansion](#stratégie-3--breakout-volatility-expansion)
   - [L'idée](#lidée-breakout)
   - [Les 4 filtres du signal](#les-4-filtres-du-signal)
   - [Entrée et Stop Loss](#entrée-et-stop-loss-breakout)
   - [Trailing Stop adaptatif (3 paliers)](#trailing-stop-adaptatif-3-paliers)
   - [Kill-Switch mensuel](#kill-switch-mensuel)
6. [Gestion du risque (Money Management)](#gestion-du-risque-money-management)
7. [Comment les stratégies cohabitent](#comment-les-stratégies-cohabitent)
8. [La boucle du bot (comment ça tourne)](#la-boucle-du-bot-comment-ça-tourne)
9. [Les fichiers et qui fait quoi](#les-fichiers-et-qui-fait-quoi)
10. [Exemple concret — Trade TREND](#exemple-concret--trade-trend)
11. [Exemple concret — Trade RANGE](#exemple-concret--trade-range)
12. [Exemple concret — Trade BREAKOUT](#exemple-concret--trade-breakout)
13. [Ce que le bot ne fait PAS](#ce-que-le-bot-ne-fait-pas)
14. [Les paramètres importants](#les-paramètres-importants-fichier-env)
15. [Infrastructure & Déploiement](#infrastructure--déploiement)

---

## L'idée générale

TradeX est un robot qui surveille le prix de cryptos **24 heures sur 24** et qui achète ou vend automatiquement quand certaines conditions sont réunies.

Le bot fonctionne sur **deux exchanges** avec **trois stratégies complémentaires** :

> **📊 Stratégie TREND** (Revolut X) : "Quand ça monte ou descend de manière confirmée, je suis la tendance."
> **🔄 Stratégie RANGE** (Revolut X + Binance) : "Quand ça n'a pas de direction claire, je joue les rebonds entre le plafond et le plancher."
> **🔥 Stratégie BREAKOUT** (Binance) : "Quand la volatilité explose et que le prix casse un niveau clé, je surfe la vague — LONG seulement."

Le bot ne devine rien. Il ne prédit pas l'avenir. Il **constate** l'état du marché et agit en conséquence.

---

## Les trois stratégies

| Stratégie | Exchange | Paires | Logique | Side |
|-----------|----------|--------|---------|------|
| 📊 **TREND** | Revolut X | 5 (BTC, SOL, XRP, LINK, SUI) | Suivi de tendance Dow Theory | Long & Short |
| 🔄 **RANGE** | Revolut X + Binance | 5 (Revolut) + 285 (Binance USDC) | Rebonds dans le range | Long & Short |
| 🔥 **BREAKOUT** | Binance | 20 (top cryptos USDC) | Casser la résistance + volatilité | **Long Only** |

---

## Stratégie 1 — Trend Following (Dow Theory)

C'est la stratégie principale. Elle est basée sur la Dow Theory, une méthode inventée il y a plus de 100 ans par Charles Dow (le créateur du Dow Jones).

### Détecter les sommets et les creux

📄 **Fichier : `swing_detector.py`**

Le bot regarde les **bougies H4** (des bougies de 4 heures) et cherche les "pics" et les "creux" du prix.

Pour trouver un sommet (swing high), il cherche une bougie dont le prix le plus haut est **supérieur** à celui des 3 bougies avant ET des 3 bougies après :

```
         ⛰️ ← sommet (les 3 bougies à gauche et à droite sont plus basses)
        / \
       /   \
      /     \
     /       \
    /         \
```

Même logique inversée pour les creux (swing low).

### Identifier la tendance

📄 **Fichier : `trend_engine.py`**

Le bot compare les sommets et creux entre eux :

### 📈 Tendance haussière (BULLISH)

Le prix fait des **escaliers qui montent** :
- Chaque sommet est **plus haut** que le précédent → **HH** (Higher High)
- Chaque creux est **plus haut** que le précédent → **HL** (Higher Low)

```
        HH ⭐
       /  \
      /    \        HH ⭐
     /      \      /  \
    /    HL ⭐\   /    \
   /         \ \ /      \
  /           \/         \
 /         HL ⭐          \
```

### 📉 Tendance baissière (BEARISH)

Le prix fait des **escaliers qui descendent** :
- Chaque sommet est **plus bas** que le précédent → **LH** (Lower High)
- Chaque creux est **plus bas** que le précédent → **LL** (Lower Low)

### ⏸️ Neutre (NEUTRAL)

Si les sommets et creux ne font pas de "beaux escaliers" → le bot dit "je ne comprends pas" → mode NEUTRAL. La **Stratégie TREND** s'arrête, et la **Stratégie RANGE** prend le relais.

### Invalidation

Le bot surveille en temps réel si la tendance est cassée :
- En **BULLISH** : si le prix descend sous le dernier HL → NEUTRAL
- En **BEARISH** : si le prix monte au-dessus du dernier LH → NEUTRAL

### Décider quand entrer (TREND)

📄 **Fichier : `strategy_trend.py`**

Le bot attend une **confirmation** avant d'entrer :

| Tendance | Signal | Condition |
|----------|--------|-----------|
| BULLISH | BUY | Prix dépasse le dernier HH + 0.2% de buffer |
| BEARISH | SELL | Prix passe sous le dernier LL - 0.2% de buffer |

**⚠️ Contrainte spot** : Revolut X ne permet pas le short selling. Les signaux SELL sont ignorés si on ne possède pas l'actif.

### Protéger ses gains (zero-risk + trailing)

📄 **Fichier : `risk_manager.py`**

**Zero-risk** : Si le prix bouge de **+2%** en faveur → le SL est déplacé au-dessus du prix d'entrée pour verrouiller **+0.5%** de profit minimum. Tu ne peux plus perdre.

**Trailing stop** : Après le zero-risk, le SL **suit le prix** à une distance de 2%. Il ne peut que monter (achat) ou descendre (vente). Il ne recule jamais.

```
Exemple achat BTC :
  Entrée : 70 000$
  +2% → 71 400$ → Zero-risk activé, SL = 70 350$ (entrée + 0.5%)
  Peak 73 000$ → Trailing SL = 71 540$ (peak × 0.98)
  Peak 74 500$ → Trailing SL = 73 010$
  Prix redescend → SL reste à 73 010$ → touché → on sort avec un gros gain 🎉
```

### Sortir du trade (TREND)

Le bot sort quand le **Stop Loss** est touché. Il n'y a **pas de Take Profit fixe** — l'idée est de laisser courir les gains tant que la tendance tient.

---

## Stratégie 2 — Mean Reversion Range

C'est la stratégie secondaire. Elle s'active **uniquement quand la tendance est NEUTRAL**, c'est-à-dire quand le prix oscille sans direction claire entre un plafond et un plancher.

### Détecter le range

📄 **Fichier : `strategy_mean_rev.py`**

Quand la tendance passe en NEUTRAL, le bot regarde les derniers niveaux clés (dernier sommet et dernier creux) pour définir un "range" — un couloir de prix :

```
──────── Range High (plafond) = dernier sommet ────────
                                                        
    Prix oscille ici ↕️       ← zone de range            
                                                        
──────── Range Low (plancher) = dernier creux ──────────
```

Le range doit avoir une **largeur minimum de 2%**. Si le plafond et le plancher sont trop proches, le bot ne trade pas (les gains potentiels seraient trop petits).

### Décider quand entrer (RANGE)

Le bot attend que le prix s'approche d'une borne du range :

| Signal | Condition | Logique |
|--------|-----------|---------|
| **BUY** | Prix ≤ Range Low × (1 + 0.2%) | "Le prix touche le plancher, il va remonter" |
| **SELL** | Prix ≥ Range High × (1 - 0.2%) | "Le prix touche le plafond, il va redescendre" |

C'est l'inverse du Trend Following : au lieu de suivre le mouvement, on **parie sur le rebond**.

### Take Profit et Stop Loss (RANGE)

Contrairement au Trend Following, les trades RANGE ont un **Take Profit fixe** :

| | Valeur | Logique |
|---|--------|---------|
| **TP** | Milieu du range | "Le prix revient au centre" |
| **SL** | Breakout au-delà de la borne opposée + 0.3% | "Le range est cassé, on coupe" |

```
Exemple SOL-USD en range 78$ – 85$ :

Signal BUY au plancher :
  Entrée : 78.16$ (78 × 1.002)
  TP : 81.50$ (milieu du range) → on vise le centre
  SL : 77.77$ (78 × 0.997) → si le prix casse le plancher, on coupe

Signal SELL au plafond :
  Entrée : 84.83$ (85 × 0.998)
  TP : 81.50$ (milieu du range)
  SL : 85.26$ (85 × 1.003)
```

### Cooldown après breakout

Si le prix **casse le range** (breakout) et que le SL est touché, le bot active un **cooldown de 3 bougies H4** (= 12 heures). Pendant ce temps, pas de nouveau trade RANGE sur cette paire.

Pourquoi ? Parce qu'un breakout signifie souvent qu'une tendance démarre. Il faut laisser le temps au marché de se stabiliser.

### Sortie forcée

Si la tendance passe de NEUTRAL à BULLISH ou BEARISH **pendant qu'un trade RANGE est ouvert**, le bot **ferme immédiatement** la position RANGE. La Stratégie TREND reprend la main.

---

## Stratégie 3 — Breakout Volatility Expansion

C'est la stratégie complémentaire aux deux précédentes. Elle tourne sur **Binance** en tant que **bot séparé** et ne gagne que quand le marché fait un mouvement directionnel puissant — exactement quand les trades RANGE perdent.

### L'idée (Breakout)

> Quand la volatilité explose et que le prix casse un niveau clé avec du volume, c'est souvent le début d'un gros mouvement. On entre et on laisse courir avec un trailing stop adaptatif.

**⚠️ LONG ONLY** : le backtest a montré que les shorts détruisent la performance. Le bot n'entre qu'en achat.

### Les 4 filtres du signal

📄 **Fichier : `breakout_detector.py`**

Un signal Breakout est généré seulement quand **4 conditions sont réunies simultanément** sur une bougie H4 :

| # | Filtre | Indicateur | Condition | Pourquoi |
|---|--------|------------|-----------|----------|
| 1 | **Cassure de prix** | Canal Donchian (20 périodes) | Close > Donchian High | Le prix dépasse le plus haut des 20 dernières bougies |
| 2 | **Volatilité en expansion** | Bandes de Bollinger (20,2) | BB Width > 1.0× moyenne | Les bandes s'écartent = la volatilité augmente |
| 3 | **Tendance confirmée** | ADX (14 périodes) | ADX > 25 | Le mouvement a de la force directionnelle |
| 4 | **Volume supérieur** | Volume vs moyenne 20 périodes | Volume > 1.2× moyenne | Le breakout est accompagné de volume |

```
Exemple : SOL-USDC, bougie H4 du 15 mars

  Close = 142$ > Donchian High (139$) ✅ Cassure
  BB Width = 0.08 > 0.06 (1.3× moyenne) ✅ Expansion
  ADX = 32 > 25 ✅ Tendance forte
  Volume = 12M > 8M (1.5× moyenne) ✅ Volume

  → 🔥 SIGNAL BREAKOUT LONG à 142$
```

### Entrée et Stop Loss (Breakout)

| | Formule | Logique |
|---|---------|---------|
| **Entrée** | Market order au prix actuel | On entre immédiatement quand le signal est détecté |
| **SL initial** | Close - 1.5 × ATR | Protection basée sur la volatilité (ATR = Average True Range) |

**Guard de sécurité** : le bot vérifie que le prix actuel est bien **au-dessus du SL** et à au moins **0.3% de distance**. Si le marché a trop bougé entre le signal et l'exécution → le trade est annulé.

### Trailing Stop adaptatif (3 paliers)

C'est le cœur de la stratégie et la raison de sa performance. Contrairement aux stratégies TREND et RANGE qui ont des TP/SL fixes, le Breakout utilise un **trailing stop qui évolue par paliers** :

```
Palier 0 — En dessous de +2% de gain
  → SL reste au SL initial (1.5×ATR sous l'entrée)

Palier 1 — À partir de +2% de gain depuis l'entrée
  → SL remonte à Entrée + 0.2% (quasi breakeven)
  → Tu ne peux quasi plus perdre

Palier 2 — À partir de +5% de gain depuis l'entrée
  → SL remonte à Entrée + 2% (profit verrouillé)
  → + Trailing ATR serré : Peak - 1.5×ATR
  → Le SL suit le prix de plus en plus près
```

**Le SL ne peut que monter, jamais descendre.** À chaque nouveau peak de prix, le SL est recalculé.

```
Exemple BTC-USDC :
  Entrée : 68 000$
  SL initial : 66 500$ (68000 - 1.5×1000)

  Peak 69 400$ (+2.1%) → Palier 1 → SL = 68 136$ (entrée + 0.2%)
  Peak 71 500$ (+5.1%) → Palier 2 → SL = max(69 360$, 71500-1500) = 70 000$
  Peak 73 000$          → SL = max(69 360$, 73000-1500) = 71 500$
  Prix redescend à 71 400$ → SL 71 500$ touché → Sortie

  Gain : (71 400 - 68 000) × 0.0073 = 24.82$ 🎉
```

### Kill-Switch mensuel

Le bot intègre un **coupe-circuit automatique** : si la performance du mois en cours atteint **-10%**, toutes les positions sont fermées et aucune nouvelle position n'est ouverte jusqu'au mois suivant.

```
Equity début mois : 2 000$
Equity actuelle : 1 780$ → perf mois = -11% < -10%
→ 🚨 KILL-SWITCH ! Fermeture de tout. Pause jusqu'au 1er du mois prochain.
```

### Résultats du backtest (12 mois, 20 paires)

| Métrique | Valeur |
|----------|--------|
| **Rendement total** | **+85.6%** |
| Profit Factor | 1.63 |
| Win Rate | 66.9% |
| Nombre de trades | 160 |
| Drawdown max | -30% |
| Sharpe Ratio | 1.44 |
| Exit via Trailing SL | 129/160 (80.6%) |

---

## Gestion du risque (Money Management)

📄 **Fichier : `risk_manager.py`**

### Risque par stratégie

Les trois stratégies n'ont **pas le même budget risque** :

| Stratégie | Exchange | Risque par trade | Max positions | Logique |
|-----------|----------|-----------------|---------------|---------|
| 📊 TREND | Revolut X | **3%** du capital | 3 | Stratégie principale, plus fiable |
| 🔄 RANGE | Revolut X / Binance | **2%** du capital | 3 | Stratégie secondaire, plus risquée |
| 🔥 BREAKOUT | Binance | **2%** du capital | 3 | Complémentaire, long only |

### Capital séparé (Binance)

Sur Binance, les bots RANGE et BREAKOUT partagent le même compte mais ont un **capital alloué virtuel** :

```
Compte Binance : 2 000 USDC total

BINANCE_RANGE_ALLOCATED_BALANCE = 1200    → bot RANGE utilise max 1200 USDC
BINANCE_BREAKOUT_ALLOCATED_BALANCE = 800  → bot BREAKOUT utilise max 800 USDC

Sécurité : chaque bot utilise min(alloué, USDC disponible réel)
```

### Plafond de risque global

Le bot impose un **risque total maximum de 6%** du capital, toutes positions confondues. Si le risque cumulé atteint 6%, plus aucune position ne peut être ouverte, quelle que soit la stratégie.

### Règles communes

| Règle | Revolut (TREND/RANGE) | Binance (RANGE) | Binance (BREAKOUT) |
|-------|----------------------|-----------------|-------------------|
| Risque par trade | 3% / 2% | 2% | 2% |
| Allocation max par position | 20% | 30% | 30% |
| Positions simultanées max | 3 | 3 | 3 |
| Plafond risque total | 6% | 6% | 6% |

### Calcul de la taille (exemple TREND)

```
Capital = 1050 USD
Risque TREND = 3% → 31.50 USD
Budget max par position = 1050 × 20% = 210 USD

Achat SOL à 90$, SL à 82$
Distance SL = 8$
Taille idéale = 31.50 / 8 = 3.94 SOL (coût 354$) > 210$ → cappé !
Taille plafonnée = 210 / 90 = 2.33 SOL
Risque réel = 2.33 × 8 = 18.67 USD (1.78% du capital)
```

### Calcul de la taille (exemple RANGE)

```
Capital = 1050 USD
Risque RANGE = 2% → 21 USD
Budget max par position = 210 USD

Achat SOL à 78.16$, SL à 77.77$
Distance SL = 0.39$
Taille idéale = 21 / 0.39 = 53.85 SOL (coût 4 208$) > 210$ → cappé !
Taille plafonnée = 210 / 78.16 = 2.69 SOL
Risque réel = 2.69 × 0.39 = 1.05 USD (0.1% du capital)
```

---

## Comment les stratégies cohabitent

### Architecture multi-bot

```
┌───────────────────────────────────────────────────────────┐
│                  REVOLUT X (5 paires USD)                  │
│                                                           │
│  📊 TREND + 🔄 RANGE → même bot, exclusivité par paire  │
│  Max 3 positions simultanées                              │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                  BINANCE (285 paires USDC)                 │
│                                                           │
│  🔄 RANGE (bot 1)     │  🔥 BREAKOUT (bot 2)             │
│  285 paires USDC      │  20 paires USDC                   │
│  Capital alloué séparé │  Capital alloué séparé            │
│  Max 3 positions       │  Max 3 positions, Long Only      │
│  Ordres OCO natifs     │  Trailing dynamique (polling)     │
└───────────────────────────────────────────────────────────┘
```

### Les garde-fous

1. **Exclusivité TREND/RANGE** : sur Revolut X, TREND et RANGE ne sont jamais actifs simultanément sur la même paire
2. **Bots indépendants** : sur Binance, les bots RANGE et BREAKOUT tournent séparément avec capital alloué distinct
3. **Sortie forcée** : si la tendance se confirme pendant un trade RANGE, le bot ferme le RANGE immédiatement
4. **Kill-Switch** : le bot BREAKOUT se coupe si le mois perd plus de 10%
5. **Complémentarité** : RANGE gagne quand le marché hésite, BREAKOUT gagne quand le marché explose → couverture mutuelle

---

## La boucle du bot (comment ça tourne)

📄 **Fichier : `bot.py`**

### ⚡ Toutes les 30 secondes (boucle rapide)

1. Demander le prix actuel de chaque paire
2. **Si position TREND ouverte** : vérifier SL, zero-risk, trailing stop
3. **Si position RANGE ouverte** : vérifier TP (milieu du range) et SL (breakout)
4. **Si pas de position** :
   - Tendance BULLISH/BEARISH → chercher signal TREND
   - Tendance NEUTRAL → chercher signal RANGE
5. Vérifier si la tendance est invalidée
6. Si tendance confirmée pendant un trade RANGE → sortie forcée

### 🕐 Toutes les 4 heures (analyse lente)

1. Récupérer les nouvelles bougies H4
2. Recalculer sommets, creux, tendance
3. Si NEUTRAL → construire/mettre à jour le range
4. Si tendance confirmée → supprimer le range
5. Mettre à jour les seuils

### Schéma simplifié

```
┌─────────────────────────────────────────────┐
│           DÉMARRAGE DU BOT                  │
│  → Charger les 100 dernières bougies H4     │
│  → Trouver les sommets/creux                │
│  → Classifier la tendance                   │
│  → Calculer les seuils + ranges             │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│      BOUCLE TOUTES LES 30 SEC               │
│                                              │
│  🔄 Pour chaque paire :                     │
│     │                                        │
│     ├─ Lire le prix actuel                   │
│     │                                        │
│     ├─ Position ouverte ?                    │
│     │   ├─ 📊 TREND :                       │
│     │   │   ├─ SL touché ? → Couper 🛑      │
│     │   │   ├─ Zero-risk ? → Ajuster SL 🔒  │
│     │   │   └─ Trailing ? → Suivre 📈       │
│     │   │                                    │
│     │   └─ 🔄 RANGE :                       │
│     │       ├─ TP touché ? → Prendre gain 🎯│
│     │       ├─ SL touché ? → Couper 🛑      │
│     │       └─ Tendance confirmée ? → Exit ⚠️│
│     │                                        │
│     └─ Pas de position ?                     │
│         ├─ BULLISH/BEARISH → Signal TREND ?  │
│         └─ NEUTRAL → Signal RANGE ?          │
│                                              │
│  💤 Dormir 30 secondes                       │
│  🔁 Recommencer                              │
└──────────────────────────────────────────────┘
```

---

## Les fichiers et qui fait quoi

| Fichier | Rôle en une phrase |
|---------|-------------------|
| `config.py` | Charge les paramètres depuis le fichier `.env` (clés API, % de risque, etc.) |
| `models.py` | Définit les "objets" : bougie, swing, tendance, ordre, position, range, stratégie |
| `swing_detector.py` | Trouve les sommets et les creux dans les bougies |
| `trend_engine.py` | Classifie la tendance : BULLISH, BEARISH ou NEUTRAL |
| `strategy_trend.py` | 📊 Stratégie Trend Following : décide quand entrer/sortir en tendance |
| `strategy_mean_rev.py` | 🔄 Stratégie Mean Reversion : gère les trades dans le range |
| `breakout_detector.py` | 🔥 Détecte les signaux Breakout (BB + Donchian + ADX + Volume) |
| `risk_manager.py` | Calcule combien on mise (risque adapté par stratégie) et gère zero-risk/trailing |
| `bot.py` | Boucle principale Revolut X (TREND + RANGE sur 5 paires) |
| `bot_binance.py` | Boucle principale Binance RANGE (285 paires USDC, ordres OCO) |
| `bot_binance_breakout.py` | 🔥 Boucle principale Binance BREAKOUT (20 paires, trailing adaptatif) |
| `revolut_client.py` | Communique avec l'API Revolut X |
| `binance_client.py` | Communique avec l'API Binance |
| `telegram.py` | Envoie les alertes sur ton téléphone via Telegram |
| `trade_logger.py` | Log chaque trade dans Firebase Firestore |

### Dashboards de monitoring

| Dashboard | Port | Exchange | Fichier |
|-----------|------|----------|---------|
| 🟡 Binance RANGE | 8503 | `binance` | `dashboard/app_binance.py` |
| 🔥 Binance BREAKOUT | 8504 | `binance-breakout` | `dashboard/app_binance_breakout.py` |

### Séparation importante

Tout le dossier `src/core/` (5 fichiers) contient la **logique pure** — les maths, les règles, les décisions. Ce code ne fait **aucun appel réseau**. Il est testable sans connexion internet.

Le dossier `src/exchange/` et `src/notifications/` gèrent les **communications** avec l'extérieur (API Revolut X, Telegram).

---

## Exemple concret — Trade TREND

Imaginons ce scénario sur SOL-USD :

### 1️⃣ Le bot détecte une tendance BULLISH

```
Analyse des 100 dernières bougies H4...
Tendance = BULLISH (HH + HL = escaliers qui montent)
Seuil d'entrée : 90$ (dernier HH)
Stop Loss : 82$ (dernier HL)
```

### 2️⃣ Le prix dépasse le seuil → Signal TREND BUY

```
Prix = 90.18$ (dépasse 90$ + 0.2%)
→ 📊🎯 Signal TREND BUY !
→ Risque 3% : 31.50 USD → Taille = 2.33 SOL (cappé à 210$)
```

### 3️⃣ Zero-risk + Trailing

```
Prix = 92$ (+2%) → 🔒 Zero-risk, SL = 90.63$
Prix = 95$ → 📈 Trailing SL = 93.10$
Prix = 98$ → 📈 Trailing SL = 96.04$
Prix = 96$ → 🛑 SL touché → Sortie à ~96$
```

**Gain : (96 - 90.18) × 2.33 = 13.56 USD** 🎉

---

## Exemple concret — Trade RANGE

Imaginons SOL-USD passe en NEUTRAL :

### 1️⃣ Le bot détecte un range

```
Tendance invalidée → NEUTRAL
Dernier sommet : 85$, Dernier creux : 78$
→ 🔄 Range détecté : 78$ – 85$ (largeur 8.97% > 2% minimum ✅)
→ Milieu du range : 81.50$
```

### 2️⃣ Le prix touche le bas du range → Signal RANGE BUY

```
Prix = 78.16$ (≤ 78 × 1.002)
→ 🔄🎯 Signal RANGE BUY !
→ Entrée : 78.16$ | TP : 81.50$ (milieu) | SL : 77.77$ (breakout bas)
→ Risque 2% : 21 USD → Taille = 2.69 SOL (cappé à 210$)
```

### 3️⃣ Scénario A — Le prix revient au milieu (TP touché)

```
Prix monte à 81.50$ → 🔄🎯 TP atteint !
→ Le bot vend au milieu du range
→ Gain : (81.50 - 78.16) × 2.69 = 8.98 USD 🎉
```

### 3️⃣ Scénario B — Le prix casse le range (SL touché)

```
Prix descend à 77.70$ → 🔄🛑 SL touché (breakout bas)
→ Le bot vend pour couper les pertes
→ Perte : (78.16 - 77.70) × 2.69 = 1.24 USD
→ ⏳ Cooldown 12h activé — pas de nouveau trade RANGE ici
```

### 3️⃣ Scénario C — La tendance se confirme (sortie forcée)

```
Pendant le trade RANGE, la prochaine analyse H4 dit : BULLISH !
→ ⚠️ Tendance confirmée → sortie forcée de la position RANGE
→ La Stratégie TREND reprend la main sur SOL-USD
```

---

## Exemple concret — Trade BREAKOUT

Imaginons ETH-USDC avec un breakout haussier :

### 1️⃣ Le bot détecte un signal Breakout

```
Bougie H4 clôturée à 2 080$ :
  Close (2 080$) > Donchian High 20p (2 050$)  ✅ Cassure
  BB Width = 0.09 > 0.07 (1.3× moy)             ✅ Expansion
  ADX = 31 > 25                                   ✅ Tendance
  Volume = 15M > 10M (1.5× moy)                   ✅ Volume

→ 🔥 SIGNAL BREAKOUT LONG !
```

### 2️⃣ Le bot vérifie et entre

```
Prix actuel ticker : 2 082$
SL signal : 2 080 - 1.5×40 = 2 020$
Guard : 2 082 > 2 020 et distance 3.0% > 0.3% ✅

→ MARKET BUY ETH @ 2 082$
→ SL initial = 2 020$ | Risque 2% = 16$ | Size = 0.17 ETH
```

### 3️⃣ Trailing adaptatif

```
Peak 2 124$ (+2.0%) → 🔒 Palier 1 → SL = 2 086$ (entrée + 0.2%)
Peak 2 190$ (+5.2%) → 🔒 Palier 2 → SL = max(2 124$, 2190-60) = 2 130$
Peak 2 250$          → SL = max(2 124$, 2250-60) = 2 190$
Prix redescend       → SL touché à 2 190$

Gain : (2 190 - 2 082) × 0.17 = 18.36$ 🎉
```

---

## Ce que le bot ne fait PAS

| ❌ Ne fait pas | ✅ Fait à la place |
|---------------|-------------------|
| Prédire l'avenir | Suivre la tendance OU jouer les rebonds selon le contexte |
| Miser tout le capital | Risquer 3% (trend) ou 2% (range) max par trade, plafond 6% |
| Shorter sans avoir l'actif | N'entre en SELL que si on possède l'actif (exchange spot) |
| Trader en permanence | TREND ou RANGE selon le contexte, neutre si rien n'est clair |
| Mélanger les stratégies | Une seule stratégie par paire à tout moment |

---

## Les paramètres importants (fichier `.env`)

### Paramètres Trend Following 📊

| Paramètre | Valeur | Ce que ça fait |
|-----------|--------|----------------|
| `RISK_PERCENT_TREND` | 3% | Risque par trade TREND |
| `ENTRY_BUFFER_PERCENT` | 0.2% | Marge de confirmation d'entrée |
| `SL_BUFFER_PERCENT` | 0.3% | Marge pour éviter les fausses sorties |
| `ZERO_RISK_TRIGGER_PERCENT` | 2% | Mouvement requis pour activer le zero-risk |
| `ZERO_RISK_LOCK_PERCENT` | 0.5% | Profit minimum verrouillé |
| `TRAILING_STOP_PERCENT` | 2% | Distance du trailing stop |

### Paramètres Mean Reversion Range 🔄

| Paramètre | Valeur | Ce que ça fait |
|-----------|--------|----------------|
| `RISK_PERCENT_RANGE` | 2% | Risque par trade RANGE |
| `RANGE_ENTRY_BUFFER_PERCENT` | 0.2% | Marge d'entrée sur les bornes du range |
| `RANGE_SL_BUFFER_PERCENT` | 0.3% | Marge du SL au-delà de la borne |
| `RANGE_WIDTH_MIN` | 2% | Largeur minimum du range pour trader |
| `RANGE_COOLDOWN_BARS` | 3 | Bougies H4 de pause après un breakout (= 12h) |

### Paramètres Breakout Volatility Expansion 🔥

| Paramètre | Valeur | Ce que ça fait |
|-----------|--------|----------------|
| `BINANCE_BREAKOUT_RISK_PERCENT` | 2% | Risque par trade Breakout |
| `BINANCE_BREAKOUT_MAX_POSITIONS` | 3 | Nombre max de trades Breakout ouverts |
| `BINANCE_BREAKOUT_BB_PERIOD` | 20 | Période des Bandes de Bollinger |
| `BINANCE_BREAKOUT_BB_STD` | 2.0 | Écart-type des BB |
| `BINANCE_BREAKOUT_BB_EXPANSION` | 1.0 | Multiplicateur d'expansion BB Width |
| `BINANCE_BREAKOUT_DONCHIAN_PERIOD` | 20 | Période du canal Donchian |
| `BINANCE_BREAKOUT_ADX_THRESHOLD` | 25 | Seuil ADX minimum |
| `BINANCE_BREAKOUT_VOL_MULT` | 1.2 | Multiplicateur volume vs moyenne |
| `BINANCE_BREAKOUT_SL_ATR_MULT` | 1.5 | Multiplicateur ATR pour le SL initial |
| `BINANCE_BREAKOUT_ADAPTIVE_TRAIL` | true | Active le trailing par paliers |
| `BINANCE_BREAKOUT_TRAIL_STEP1_PCT` | 2% | Gain requis pour Palier 1 |
| `BINANCE_BREAKOUT_TRAIL_STEP2_PCT` | 5% | Gain requis pour Palier 2 |
| `BINANCE_BREAKOUT_TRAIL_LOCK1_PCT` | 0.2% | Profit verrouillé Palier 1 |
| `BINANCE_BREAKOUT_TRAIL_LOCK2_PCT` | 2% | Profit verrouillé Palier 2 |
| `BINANCE_BREAKOUT_KILL_SWITCH` | true | Active le kill-switch mensuel |
| `BINANCE_BREAKOUT_KILL_PCT` | -10% | Seuil du kill-switch |

### Paramètres globaux

| Paramètre | Valeur | Ce que ça fait |
|-----------|--------|----------------|
| `MAX_TOTAL_RISK_PERCENT` | 6% | Plafond de risque total (toutes positions) |
| `MAX_POSITION_PERCENT` | 20-30% | Part max du capital par position |
| `MAX_SIMULTANEOUS_POSITIONS` | 3 | Nombre max de trades ouverts par bot |
| `SWING_LOOKBACK` | 3 | Bougies de confirmation pour les sommets/creux |
| `POLLING_INTERVAL_SECONDS` | 30s | Fréquence de vérification du prix |
| `TRADING_PAIRS` | BTC, SOL, XRP, LINK, SUI | Cryptos Revolut X (5 paires USD) |
| `BINANCE_BREAKOUT_PAIRS` | 20 paires USDC | Cryptos Breakout (BTC, ETH, SOL, etc.) |

---

## Infrastructure & Déploiement

### VPS (Contabo)

| | Détail |
|---|--------|
| **OS** | Ubuntu 22.04 LTS |
| **IP** | 213.199.41.168 |
| **Connexion** | `ssh BOT-VPS` |
| **App** | `/opt/tradex` |
| **Python** | 3.10, venv `.venv` |
| **Gestion** | systemd services |

### Services actifs

| Service | Description | Port |
|---------|-------------|------|
| `tradex` | Bot Revolut X (TREND + RANGE, 5 paires) | — |
| `tradex-binance` | Bot Binance RANGE (285 paires USDC) | — |
| `tradex-binance-breakout` | Bot Binance BREAKOUT (20 paires, Long Only) | — |
| `tradex-binance-dashboard` | Dashboard Streamlit RANGE | 8503 |
| `tradex-binance-breakout-dashboard` | Dashboard Streamlit BREAKOUT | 8504 |

### Commandes utiles

```bash
# Logs en direct
ssh BOT-VPS 'sudo journalctl -u tradex-binance-breakout -f'

# État des services
ssh BOT-VPS 'for svc in tradex tradex-binance tradex-binance-breakout; do echo -n "$svc: "; sudo systemctl is-active $svc; done'

# Déployer
bash deploy/deploy-binance-breakout.sh

# Dashboards
# RANGE    : http://213.199.41.168:8503
# BREAKOUT : http://213.199.41.168:8504
```
