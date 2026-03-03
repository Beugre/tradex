# 🧠 TradeX — Comment ça marche (version simple)

## Table des matières

1. [L'idée générale](#lidée-générale)
2. [Les quatre bots](#les-quatre-bots)
3. [Bot 1 — Trail Range (Binance)](#bot-1--trail-range-binance)
   - [Détecter le range](#détecter-le-range)
   - [Décider quand entrer](#décider-quand-entrer-range)
   - [OCO : TP + SL en un seul ordre](#oco--tp--sl-en-un-seul-ordre)
   - [Trail@TP : le trailing OCO](#trailtp--le-trailing-oco)
   - [Sortie forcée si tendance confirmée](#sortie-forcée-si-tendance-confirmée)
4. [Bot 2 — CrashBot (Binance)](#bot-2--crashbot-binance)
   - [L'idée](#lidée-crashbot)
   - [Détecter un crash](#détecter-un-crash)
   - [Entrée et Stop Loss](#entrée-et-stop-loss-crashbot)
   - [Step Trailing : le TP qui monte par paliers](#step-trailing--le-tp-qui-monte-par-paliers)
   - [Kill-Switch mensuel](#kill-switch-mensuel)
5. [Bot 3 — Momentum Continuation (Revolut X)](#bot-3--momentum-continuation-revolut-x)
   - [L'idée](#lidée-momentum)
   - [Phase 1 : Filtre macro M15](#phase-1--filtre-macro-m15)
   - [Phase 2 : Détecter l'impulsion M5](#phase-2--détecter-limpulsion-m5)
   - [Phase 3 : Attendre le pullback](#phase-3--attendre-le-pullback)
   - [Phase 4 : Entrée sur reprise](#phase-4--entrée-sur-reprise)
   - [Exécution maker-only](#exécution-maker-only)
6. [Bot 4 — Infinity Bot (Revolut X)](#bot-4--infinity-bot-revolut-x)
   - [L'idée](#lidée-infinity)
   - [Trailing High](#trailing-high--le-prix-de-référence-dynamique)
   - [Paliers d'achat](#paliers-dachat-dca-inversé)
   - [Paliers de vente](#paliers-de-vente-distribution-progressive)
   - [Sécurités](#sécurités)
7. [Allocation dynamique du capital](#allocation-dynamique-du-capital)
   - [Comment ça marche](#comment-ça-marche)
   - [Pourquoi cette logique](#pourquoi-cette-logique)
8. [Gestion du risque (Money Management)](#gestion-du-risque-money-management)
9. [La boucle de chaque bot](#la-boucle-de-chaque-bot)
10. [Les fichiers et qui fait quoi](#les-fichiers-et-qui-fait-quoi)
11. [Exemple concret — Trade RANGE](#exemple-concret--trade-range)
12. [Exemple concret — Trade CRASH](#exemple-concret--trade-crash)
13. [Exemple concret — Trade MOMENTUM](#exemple-concret--trade-momentum)
14. [Ce que les bots ne font PAS](#ce-que-les-bots-ne-font-pas)
15. [Les paramètres importants](#les-paramètres-importants-fichier-env)
16. [Infrastructure & Déploiement](#infrastructure--déploiement)

---

## L'idée générale

TradeX est un **écosystème de 4 bots** qui surveillent le marché crypto **24h/24** et tradent automatiquement quand les conditions sont réunies.

Les bots fonctionnent sur **2 exchanges** avec **4 stratégies complémentaires** :

> **🔄 Trail Range** (Binance) : "Quand le prix oscille dans un couloir, je joue les rebonds entre le plafond et le plancher."
> **💥 CrashBot** (Binance) : "Quand une crypto s'effondre brutalement, j'achète le dip et je laisse remonter."
> **🚀 Momentum** (Revolut X) : "Quand une bougie M5 explose avec du volume, j'attends le pullback puis j'entre dans la direction."
> **♾️ Infinity** (Revolut X) : "Quand BTC baisse de 5%+, j'accumule par paliers DCA puis je revends progressivement."

Aucun bot ne prédit l'avenir. Chacun **constate** un pattern spécifique et agit en conséquence.

---

## Les quatre bots

| Bot | Exchange | Paires | Logique | Side | Capital |
|-----|----------|--------|---------|------|--------|
| 🔄 **Trail Range** | Binance (USDC) | ~284 (auto-discovery) | Rebonds dans le range + trailing OCO | Long & Short | 10–40% Binance (dynamique) |
| 💥 **CrashBot** | Binance (USDC) | ~284 (auto-discovery) | Dip-buy + step-trail | **Long Only** | 60–90% Binance (dynamique) |
| 🚀 **Momentum** | Revolut X (USD) | 7 (ETH, SOL, BNB, XRP, LINK, ADA, LTC) | Impulsion M5 → pullback → entrée | **Long Only** | 35% Revolut X |
| ♾️ **Infinity** | Revolut X (USD) | BTC-USD uniquement | DCA inversé + vente paliers | **Long Only** | 65% Revolut X |

---

## Bot 1 — Trail Range (Binance)

C'est le bot **mean-reversion**. Il trade quand le prix oscille sans direction claire entre un plafond et un plancher.

### Détecter le range

📄 **Fichier : `strategy_mean_rev.py`**

Le bot analyse les bougies H4 pour trouver les swings (sommets et creux). Quand la tendance est **NEUTRAL** (pas d'escaliers clairs ni à la hausse ni à la baisse), il définit un range :

```
──────── Range High (plafond) = dernier sommet ────────

    Prix oscille ici ↕️       ← zone de range

──────── Range Low (plancher) = dernier creux ──────────
```

Le range doit avoir une **largeur minimum de 2%**. Si le corridor est trop étroit, le bot ne trade pas.

### Détecter la tendance

📄 **Fichiers : `swing_detector.py` + `trend_engine.py`**

Le bot utilise la **Dow Theory** pour classifier la tendance :

- **BULLISH** : sommets de plus en plus hauts (HH) + creux de plus en plus hauts (HL) → escaliers montants
- **BEARISH** : sommets de plus en plus bas (LH) + creux de plus en plus bas (LL) → escaliers descendants
- **NEUTRAL** : pas de pattern clair → **c'est là que le bot RANGE entre en jeu**

### Décider quand entrer (RANGE)

Le bot attend que le prix touche une borne du range :

| Signal | Condition | Logique |
|--------|-----------|---------|
| **BUY** | Prix ≤ Range Low × (1 + 0.2%) | "Le prix touche le plancher, il va remonter" |
| **SELL** | Prix ≥ Range High × (1 - 0.2%) | "Le prix touche le plafond, il va redescendre" |

C'est de la **mean-reversion** : on parie que le prix revient au centre du range.

### OCO : TP + SL en un seul ordre

📄 **Fichier : `bot_binance.py`**

Binance supporte les **ordres OCO** (One-Cancels-Other) : un Take Profit ET un Stop Loss sont posés simultanément. Quand l'un est touché, l'autre est automatiquement annulé.

| | Valeur | Logique |
|---|--------|---------|
| **TP** | Milieu du range | "Le prix revient au centre" |
| **SL** | Breakout au-delà de la borne + marge | "Le range est cassé, on coupe" |

### Trail@TP : le trailing OCO

Quand le prix s'approche du TP (à moins de `SWAP_PCT`), au lieu de prendre le gain, le bot **swap l'OCO** :

1. Annuler l'OCO actuel
2. Nouveau SL = ancien TP × (1 - 2%) → le gain est verrouillé
3. Nouveau TP = ancien TP × (1 + 1%) → on vise plus haut
4. Poser un nouvel OCO avec ces niveaux

Ce mécanisme peut se répéter plusieurs fois (step 1, step 2, step 3...). Chaque step verrouille un palier de profit supplémentaire.

```
Exemple BUY SOL-USDC dans un range 78$ – 85$ :

  Entrée : 78.16$ | TP initial : 81.50$ (milieu) | SL : 77.77$

  Prix monte à ~81.40$ → Trail swap step 1 !
    Nouveau SL = 79.87$ (81.50 × 0.98) → profit verrouillé
    Nouveau TP = 82.32$ (81.50 × 1.01)

  Prix monte à ~82.20$ → Trail swap step 2 !
    Nouveau SL = 80.67$ (82.32 × 0.98)
    Nouveau TP = 83.14$ (82.32 × 1.01)

  Prix redescend → SL touché à 80.67$
  Gain : (80.67 - 78.16) × size = profit 🎉
```

### Sortie forcée si tendance confirmée

Si pendant un trade RANGE, la tendance passe de NEUTRAL à BULLISH ou BEARISH → le bot **ferme immédiatement** la position. Le range n'est plus valide.

### Cooldown après breakout

Si le SL est touché (le prix casse le range), le bot active un **cooldown de 3 bougies H4** (= 12h). Pas de nouveau trade sur cette paire pendant ce temps.

---

## Bot 2 — CrashBot (Binance)

### L'idée (CrashBot)

> Quand une crypto perd **20% ou plus en 48 heures**, c'est souvent une surréaction. Le bot achète le dip et laisse le prix remonter avec un trailing par paliers.

**⚠️ LONG ONLY** : le bot n'achète que les baisses. Pas de short.

### Détecter un crash

📄 **Fichier : `crashbot_detector.py`**

Le bot compare le prix actuel à celui d'il y a **12 bougies H4** (48 heures) :

```
Prix il y a 48h : 100$
Prix maintenant : 78$
Drop = -22% → ≥ 20% de seuil → 💥 SIGNAL CRASH !
```

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `drop_threshold` | 20% | Baisse minimum pour déclencher |
| `lookback_bars` | 12 | 12 × 4h = 48 heures de recul |

### Entrée et Stop Loss (CrashBot)

| | Formule | Logique |
|---|---------|---------|
| **Entrée** | Market order au prix actuel | On achète immédiatement le dip |
| **TP initial** | Entrée × (1 + 8%) | Objectif : +8% de rebond |
| **SL** | Entrée - 1.5 × ATR | Protection basée sur la volatilité |

Si l'ATR n'est pas disponible, le SL est fixé à -2% sous l'entrée.

### Step Trailing : le TP qui monte par paliers

C'est le cœur de la stratégie CrashBot. Quand le prix approche du TP, au lieu de vendre :

1. Le **SL remonte au niveau du TP actuel** → profit verrouillé
2. Le **TP monte d'un step** (+0.5%)
3. Le processus se répète tant que le prix continue de monter

```
Entrée : 78$ | TP₀ = 84.24$ (+8%) | SL = 73$ (1.5×ATR)

  Prix monte à 84.20$ → TP₀ quasi touché !
    SL₁ = 84.24$ (verrouillé au TP₀) → plus de perte possible
    TP₁ = 84.63$ (+0.5%)

  Prix continue à 84.60$ →
    SL₂ = 84.63$
    TP₂ = 85.02$

  Prix redescend → SL₂ touché à 84.63$
  Gain : (84.63 - 78) × size = gros profit 🎉
```

Le trailing peut monter **indéfiniment** tant que le prix suit. Le SL ne descend jamais.

### Kill-Switch mensuel

Si la performance du mois atteint **-10%**, toutes les positions sont fermées et le bot se met en pause jusqu'au mois suivant.

```
Equity début mois : 2 000$
Equity actuelle : 1 780$ → perf = -11% < -10%
→ 🚨 KILL-SWITCH ! Pause jusqu'au 1er du mois prochain.
```

---

## Bot 3 — Momentum Continuation (Revolut X)

### L'idée (Momentum)

> Quand une bougie M5 explose avec beaucoup de volume (impulsion), le prix fait souvent un petit repli (pullback) avant de continuer dans la même direction. Le bot entre sur le pullback.

C'est un signal en **4 phases** :

### Phase 1 : Filtre macro M15

📄 **Fichier : `momentum_engine.py`**

Le bot vérifie d'abord que le marché est **actif** sur M15 :
- ATR(14) > moyenne mobile de l'ATR → la volatilité est au-dessus de la normale
- Volume > moyenne mobile du volume → le marché est liquide

Si le marché est "mort" (faible volatilité, faible volume), le bot ne cherche pas de signal.

### Phase 2 : Détecter l'impulsion M5

Le bot cherche une bougie M5 exceptionnelle qui réunit **4 critères** :

| # | Critère | Condition | Pourquoi |
|---|---------|-----------|----------|
| 1 | **Body important** | Body ≥ 0.4% du prix | Bougie avec du contenu, pas juste une mèche |
| 2 | **Volume explosif** | Volume ≥ 2× MA20 | Beaucoup plus d'échanges que la normale |
| 3 | **Close dans le top** | Close dans le top 20% de la bougie | Le prix a clôturé dans la direction du move |
| 4 | **Tendance directionnelle** | ADX(14) > 15 | Le mouvement a de la force, pas du bruit |

```
Exemple ETH-USD, bougie M5 :
  Open: 2 100$ → Close: 2 112$ (body +0.57%) ✅
  Volume: 15M (moy 20: 6M → 2.5×) ✅
  Close dans top 20% de la bougie ✅
  ADX = 22 > 15 ✅
  → 🚀 IMPULSION HAUSSIÈRE détectée !
```

### Phase 3 : Attendre le pullback

Après l'impulsion, le bot attend un **retracement sain** :

| Critère | Condition | Pourquoi |
|---------|-----------|----------|
| Retracement | 25–55% du move d'impulsion | Ni trop peu (pas de pullback), ni trop (reversal) |
| RSI(14) | Entre 40 et 65 | Pas suracheté, pas survendu |
| Prix touche EMA20 | ± tolérance | Le prix revient sur la moyenne rapide |

Le bot attend **max 35 bougies M5** (~3h) après l'impulsion. Si aucun pullback valide → signal annulé.

```
Impulsion haussière de 2 100$ à 2 112$ (move = 12$)

  Pullback à 2 108$ → retracement = 4/12 = 33% ✅
  RSI = 52 (entre 40-65) ✅
  Prix proche de EMA20 ✅
  → Pullback validé !
```

### Phase 4 : Entrée sur reprise

Le bot attend une **bougie de reprise** dans la direction de l'impulsion :
- Volume > MA10
- La bougie confirme la direction (close > open pour un long)

| | Valeur | Logique |
|---|--------|---------|
| **Entrée** | Ordre limit maker | 0% de frais sur Revolut X |
| **SL** | Sous le creux du pullback + marge | Si le pullback n'a pas tenu |
| **Risque** | 4% du capital | MC_RISK_PERCENT |

### Exécution maker-only

📄 **Fichier : `revolut_client.py`**

Revolut X facture **0% en maker** et 0.09% en taker. Le bot place donc un **ordre limit** (maker) pour ne pas payer de frais.

Si l'ordre n'est pas rempli dans les 60 secondes (`MC_MAKER_WAIT_SECONDS`), le bot peut fallback en taker (0.09%).

---

## Bot 4 — Infinity Bot (Revolut X)

C'est le bot **DCA inversé** sur BTC uniquement. Il achète les baisses par paliers et revend progressivement quand le prix remonte.

### L'idée {#lidée-infinity}

> "Quand BTC baisse de 5% par rapport à son plus haut récent, j'achète. S'il continue de baisser, j'achète encore à -10%, -15%, -20%, -25%. Puis je revends par paliers quand il remonte."

C'est l'opposé d'un grid bot : au lieu d'acheter et vendre sur des niveaux fixes, le prix de référence suit dynamiquement le marché (trailing high sur 12 jours).

### Trailing High — le prix de référence dynamique

📄 **Fichier : `infinity_engine.py`**

Le bot calcule en permanence le **plus haut des 72 dernières bougies H4** (≈ 12 jours). C'est le "trailing high" :

```
│    ╱★ Trailing High (plus haut sur 12 jours)
│   ╱  ╲
│  ╱    ╲
│ ╱      ╲──── Prix actuel
│╱         ╲
│            ╲── Drop ≥ 5% → Premier achat !
```

Quand le prix **chute de ≥ 5%** par rapport à ce trailing high → le bot déclenche le premier achat.

### Paliers d'achat (DCA inversé)

Le bot a 5 niveaux d'achat, chacun plus profond que le précédent :

| Palier | Drop depuis le référence | % du capital | RSI gate |
|--------|--------------------------|-------------|----------|
| L1 | -5% | 25% | RSI < 50 → full, 30-50 → demi |
| L2 | -10% | 20% | idem |
| L3 | -15% | 15% | idem |
| L4 | -20% | 10% | idem |
| L5 | -25% | Reste disponible | idem |

**Sizing décroissant** : on investit plus au début (quand le prix est encore proche du haut) et moins quand on est profond dans le dip. Cela évite de surcharger dans les crashs imprévus.

### Paliers de vente (distribution progressive)

Quand le prix remonte au-dessus du **PMP (Prix Moyen Pondéré)**, le bot vend 20% à chaque palier :

| Palier | Distance du PMP | Action |
|--------|-----------------|--------|
| TP1 | +0.8% | Vend 20% + active le breakeven stop |
| TP2 | +1.5% | Vend 20% |
| TP3 | +2.2% | Vend 20% |
| TP4 | +3.0% | Vend 20% |
| TP5 | +4.0% | Vend tout le reste |

### Sécurités

- **Breakeven stop** : Après TP1, si le prix retombe au PMP → vente totale (pas de perte)
- **Stop-loss** : Si le prix baisse de **-15%** sous le PMP → vente market (taker 0.09%)
- **Override sell** : Si le prix monte de +20% au-dessus du PMP → vente totale immédiate
- **Max investi** : 70% du capital alloué maximum par cycle
- **Pas de RSI gate sur les ventes** (clé de performance — +107% vs +50% avec)

### Capital et allocation {#allocation-revolut}

Le capital Revolut X est partagé entre **Infinity** et **Momentum** :

| Bot | % du capital Revolut X |
|-----|------------------------|
| ♾️ Infinity | **65%** |
| 🚀 Momentum | **35%** |

### Résultats backtest (6 ans, 2020-2026)

```
Rendement : +107.56%
Sharpe    : 3.85
PF        : 3.10
Max DD    : -15.78%
Win Rate  : 90% (18/20 trades gagnants)
Stops     : seulement 2 en 6 ans
```

---

## Allocation dynamique du capital

📄 **Fichier : `allocator.py`**

### Comment ça marche

Les bots **Trail Range** et **CrashBot** partagent le même capital sur Binance. La répartition est recalculée **1×/jour** automatiquement.

Le **Profit Factor (PF)** du Trail Range sur 90 jours détermine qui reçoit combien :

```
PF = somme des gains / |somme des pertes|

PF > 1 → le bot gagne plus qu'il ne perd
PF < 1 → le bot perd plus qu'il ne gagne
```

| PF Trail Range (90j) | Trail Range | CrashBot | Régime |
|----------------------|-------------|----------|--------|
| PF < 0.9 OU < 20 trades | **10%** | **90%** | 🛡️ DÉFENSIF |
| 0.9 ≤ PF ≤ 1.1 | **20%** | **80%** | ⚖️ NEUTRE |
| PF > 1.1 | **40%** | **60%** | 🚀 AGRESSIF |

```
Exemple : Capital Binance = 3 226 USDC, PF Trail = 0.47

  PF 0.47 < 0.9 → Régime DÉFENSIF
  Trail Range : 10% = 323 USDC
  CrashBot    : 90% = 2 904 USDC
```

### Pourquoi cette logique

- **CrashBot domine par défaut** car sa stratégie (dip-buy) est plus robuste en marché latéral
- Quand le Trail Range **prouve** qu'il gagne (PF > 1.1), on lui donne plus de capital
- Quand il perd, on réduit son exposition et CrashBot prend le relais

### Les bots Revolut X (Momentum + Infinity)

Les bots Momentum et Infinity partagent le capital Revolut X avec une allocation fixe : **65% Infinity** / **35% Momentum**. Ils sont **indépendants** de l'allocation Binance — les deux exchanges sont complètement séparés.

---

## Gestion du risque (Money Management)

📄 **Fichier : `risk_manager.py`**

### Formule commune

```python
risk_amount = capital × risk_percent     # ex: 3000 × 0.05 = 150 USDC
sl_distance = |entry - sl_price|         # ex: |78 - 73| = 5 USDC
position_size = risk_amount / sl_distance # ex: 150 / 5 = 30 unités

# Plafond : position_size × entry ≤ capital × max_position_pct
```

### Risque par bot

| Bot | Exchange | Risque/trade | Max positions | Max % capital/position |
|-----|----------|-------------|---------------|------------------------|
| 🔄 Trail Range | Binance | 5% | 3 | 30% |
| 💥 CrashBot | Binance | 2% | 3 | 30% |
| 🚀 Momentum | Revolut X | 4% | 3 | 90% |
| ♾️ Infinity | Revolut X | 15% (SL) | 1 cycle | 70% (max investi) |

### Calcul d'equity

Le capital de chaque bot est calculé en prenant le solde fiat + la valeur des positions ouvertes. L'allocateur recalcule sur l'equity totale Binance, puis répartit entre Trail Range et CrashBot.

---

## La boucle de chaque bot

### 🔄 Trail Range (`bot_binance.py`)

```
┌──────────────────────────────────────────────────────────┐
│ TOUTES LES 30 SECONDES                                  │
│                                                          │
│  Pour chaque paire (~284) :                              │
│    ├─ OCO actif ? → Vérifier si TP ou SL fill            │
│    │   ├─ TP fill → clôturer position, log Firebase      │
│    │   └─ SL fill → clôturer + cooldown 12h              │
│    ├─ Position sans OCO ? → Tenter de poser l'OCO        │
│    ├─ Près du TP ? → Trail swap OCO (step suivant)       │
│    └─ Pas de position ? → Chercher signal RANGE          │
│                                                          │
│ TOUTES LES 4 HEURES (nouvelle bougie H4)                 │
│    ├─ Recalculer swings + tendance                       │
│    ├─ NEUTRAL → construire/mettre à jour le range        │
│    └─ BULLISH/BEARISH → fermer trade RANGE si besoin     │
│                                                          │
│ 1×/JOUR                                                  │
│    └─ Recalculer l'allocation dynamique                  │
└──────────────────────────────────────────────────────────┘
```

### 💥 CrashBot (`bot_binance_crashbot.py`)

```
┌──────────────────────────────────────────────────────────┐
│ TOUTES LES 30 SECONDES                                  │
│                                                          │
│  Pour chaque paire (~284) :                              │
│    ├─ Position ouverte ? → Gérer step trailing           │
│    │   ├─ Prix ≥ trigger TP → SL monte, TP monte         │
│    │   └─ SL touché → clôturer                           │
│    └─ Pas de position ? → Chercher crash                 │
│        ├─ Drop ≥ 20% en 48h ? → 💥 SIGNAL               │
│        └─ Market BUY + OCO (TP/SL)                       │
│                                                          │
│ 1×/JOUR                                                  │
│    └─ Recalculer l'allocation dynamique                  │
│                                                          │
│ KILL-SWITCH : perf mois < -10% → stop tout              │
└──────────────────────────────────────────────────────────┘
```

### 🚀 Momentum (`bot_momentum.py`)

```
┌──────────────────────────────────────────────────────────┐
│ TOUTES LES 30 SECONDES                                  │
│                                                          │
│  Pour chaque paire (7) :                                 │
│    ├─ Récupérer bougies M5 + M15                         │
│    ├─ Filtre macro M15 actif ? → Continuer               │
│    ├─ Impulsion M5 détectée ? → Attendre pullback        │
│    ├─ Pullback valide ? → Attendre reprise               │
│    ├─ Reprise confirmée ? → 🚀 SIGNAL MOMENTUM          │
│    │   └─ Placer ordre limit maker (0% frais)            │
│    └─ Position ouverte ? → Gérer SL/trailing             │
│                                                          │
│ TOUTES LES 10 MINUTES                                    │
│    └─ Heartbeat Firebase + calcul equity                 │
└──────────────────────────────────────────────────────────┘
```

---

## Les fichiers et qui fait quoi

| Fichier | Rôle en une phrase |
|---------|-------------------|
| `config.py` | Charge les paramètres depuis `.env` (clés API, % de risque, etc.) |
| `models.py` | Définit les "objets" : bougie, swing, tendance, ordre, position, range |
| `swing_detector.py` | Trouve les sommets et les creux dans les bougies |
| `trend_engine.py` | Classifie la tendance : BULLISH, BEARISH ou NEUTRAL |
| `strategy_mean_rev.py` | 🔄 Stratégie Mean Reversion : signaux BUY/SELL dans le range |
| `strategy_trend.py` | Stratégie Trend Following : signaux BUY/SELL en tendance |
| `crashbot_detector.py` | 💥 Détecte les crashes (drop ≥ 20%) + step trailing |
| `momentum_engine.py` | 🚀 Détecte les impulsions M5, pullbacks et reprises |
| `allocator.py` | Répartit le capital Binance entre Trail Range et CrashBot |
| `risk_manager.py` | Calcul de taille de position, zero-risk, trailing, equity |
| `position_store.py` | Sérialisation/désérialisation des positions en JSON |
| `bot_binance.py` | 🔄 Boucle principale Trail Range (Binance, ~284 paires, OCO) |
| `bot_binance_crashbot.py` | 💥 Boucle principale CrashBot (Binance, dip-buy, step-trail) |
| `bot_momentum.py` | 🚀 Boucle principale Momentum (Revolut X, 7 paires, maker-only) |
| `bot.py` | (legacy) Ancien bot Dow Theory Revolut X |
| `binance_client.py` | Communique avec l'API Binance (OCO, market, balances) |
| `revolut_client.py` | Communique avec l'API Revolut X (Ed25519, limit orders) |
| `data_provider.py` | Récupère les bougies OHLCV |
| `telegram.py` | Envoie les alertes sur ton téléphone via Telegram |
| `trade_logger.py` | Log chaque trade, heartbeat et allocation dans Firebase |
| `client.py` (firebase) | Connexion Firestore, CRUD générique |

### Dashboard

| Dashboard | Port | Description |
|-----------|------|-------------|
| 📊 Dashboard unifié | 8502 | 4 onglets : Overview, Trail Range, CrashBot, Momentum |

---

## Exemple concret — Trade RANGE

Imaginons SOL-USDC en range 78$ – 85$ :

### 1️⃣ Le bot détecte un range

```
Tendance = NEUTRAL (pas d'escaliers clairs)
Dernier sommet : 85$ | Dernier creux : 78$
→ 🔄 Range détecté : 78$ – 85$ (largeur 8.97% > 2% minimum ✅)
→ Milieu du range : 81.50$
```

### 2️⃣ Le prix touche le bas → Signal BUY

```
Prix = 78.16$ (≤ 78 × 1.002)
→ 🔄 Signal RANGE BUY !
  Entrée : 78.16$ | TP : 81.50$ (milieu) | SL : 77.77$
  Risque 5% → Taille calculée par risk_manager
→ Ordre MARKET BUY exécuté
→ OCO posé : TP=81.50$ + SL=77.77$
```

### 3️⃣ Le prix remonte → Trail@TP

```
Prix monte à 81.40$ → Proche du TP !
→ 🔄 Trail swap step 1 :
    SL → 79.87$ (verrouillé) | TP → 82.32$

Prix monte à 82.20$ → Trail swap step 2 :
    SL → 80.67$ | TP → 83.14$

Prix redescend → SL touché à 80.67$
→ Gain : (80.67 - 78.16) × size 🎉
```

---

## Exemple concret — Trade CRASH

Imaginons AVAX-USDC qui crash :

### 1️⃣ Le bot détecte un crash

```
Prix il y a 48h : 28.50$
Prix maintenant : 22.20$
Drop = -22.1% → ≥ 20% → 💥 SIGNAL CRASH !
```

### 2️⃣ Le bot achète le dip

```
→ MARKET BUY AVAX @ 22.20$
  TP₀ = 23.98$ (+8%)
  SL = 20.50$ (22.20 - 1.5 × 1.13 ATR)
  Risque 2% du capital alloué
```

### 3️⃣ Step trailing

```
Prix monte à 23.95$ → Proche du TP₀ !
  SL₁ = 23.98$ (verrouillé au TP₀) → plus de perte possible
  TP₁ = 24.09$ (+0.5%)

Prix monte à 24.07$ →
  SL₂ = 24.09$
  TP₂ = 24.20$

Prix monte à 24.18$ →
  SL₃ = 24.20$
  TP₃ = 24.31$

Prix redescend → SL₃ touché à 24.20$
→ Gain : (24.20 - 22.20) × size = gros profit 🎉
```

---

## Exemple concret — Trade MOMENTUM

Imaginons ETH-USD sur Revolut X :

### 1️⃣ Filtre macro M15 OK

```
ATR(14) M15 = 18.5 > MA(ATR) 14.2 ✅ Volatilité active
Volume M15 = 12.3M > MA(Vol) 8.1M ✅ Marché liquide
→ Filtre macro passé
```

### 2️⃣ Impulsion M5 détectée

```
Bougie M5 : Open 2 100$ → Close 2 112.40$ (+0.59%)
  Body ≥ 0.4% ✅ | Volume 2.8× MA20 ✅ | Close top 20% ✅ | ADX 24 > 15 ✅
→ 🚀 IMPULSION HAUSSIÈRE | Move = +12.40$
```

### 3️⃣ Pullback validé

```
5 bougies plus tard : prix redescend à 2 108.30$
  Retracement = 4.10 / 12.40 = 33% (entre 25-55%) ✅
  RSI(14) = 51 (entre 40-65) ✅
  Prix proche de EMA20 ✅
→ Pullback valide
```

### 4️⃣ Entrée sur reprise

```
Bougie de reprise : Close 2 110.50$ > Open 2 108.30$, volume > MA10
→ 🚀 SIGNAL MOMENTUM BUY !
  Ordre limit maker @ 2 110.50$ (0% frais)
  SL = 2 106.20$ (sous le creux du pullback)
  Risque 4% de 500$ = 20$
  Taille = 20 / (2110.50 - 2106.20) = 4.65 ETH... cappé à 90% = 0.213 ETH
```

---

## Ce que les bots ne font PAS

| ❌ Ne fait pas | ✅ Fait à la place |
|---------------|-------------------|
| Prédire l'avenir | Constater des patterns (range, crash, impulsion) et réagir |
| Miser tout le capital | Risk 2–5% par trade, allocation dynamique |
| Shorter en spot | CrashBot et Momentum = long only. Trail Range = long & short |
| Trader en permanence | Chaque bot attend SES conditions spécifiques |
| Ignorer les pertes | Kill-switch mensuel, SL sur chaque trade, trailing lock |

---

## Les paramètres importants (fichier `.env`)

### Trail Range 🔄

| Paramètre | Valeur | Ce que ça fait |
|-----------|--------|----------------|
| `RISK_PERCENT` | 5% | Risque par trade Range |
| `RANGE_WIDTH_MIN` | 2% | Largeur minimum du range pour trader |
| `RANGE_COOLDOWN_BARS` | 3 | Bougies H4 de pause après un breakout (=12h) |
| `BINANCE_RANGE_TRAIL_SWAP_PCT` | variable | Distance au TP pour déclencher le swap OCO |
| `BINANCE_RANGE_TRAIL_STEP_PCT` | ~1% | Extension du TP par step |
| `BINANCE_RANGE_TRAIL_SL_LOCK_PCT` | ~2% | Protection du profit verrouillé |

### CrashBot 💥

| Paramètre | Valeur | Ce que ça fait |
|-----------|--------|----------------|
| `drop_threshold` | 20% | Baisse minimum pour déclencher un signal |
| `lookback_bars` | 12 | Fenêtre = 12 × 4h = 48 heures |
| `tp_pct` | 8% | Take profit initial |
| `atr_sl_mult` | 1.5 | Multiplicateur ATR pour le SL |
| `trail_step_pct` | 0.5% | Extension du TP par step |
| `BINANCE_CRASHBOT_KILL_PCT` | -10% | Seuil du kill-switch mensuel |

### Momentum 🚀

| Paramètre | Valeur | Ce que ça fait |
|-----------|--------|----------------|
| `MC_RISK_PERCENT` | 4% | Risque par trade |
| `MC_MAX_POSITIONS` | 3 | Nombre max de trades ouverts |
| `MC_MAX_POSITION_PCT` | 90% | Part max du capital par position |
| `MC_POLLING_SECONDS` | 30s | Fréquence de vérification |
| `MC_MAKER_WAIT_SECONDS` | 60s | Temps d'attente pour un fill maker |
| `impulse_body_min_pct` | 0.4% | Body minimum de l'impulsion |
| `impulse_vol_mult` | 2× | Volume minimum (vs MA20) |
| `pullback_retrace_min/max` | 25–55% | Fenêtre de retracement valide |
| `adx_min` | 15 | ADX minimum pour confirmer la direction |

### Allocation dynamique

| Paramètre | Valeur | Ce que ça fait |
|-----------|--------|----------------|
| `PF_LOW` | 0.9 | Seuil bas du PF (en-dessous → Défensif) |
| `PF_HIGH` | 1.1 | Seuil haut du PF (au-dessus → Agressif) |
| `MIN_TRADES` | 20 | Nombre minimum de trades pour évaluer le PF |

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
| `tradex-binance` | Bot Trail Range (284 paires USDC, OCO) | — |
| `tradex-binance-crashbot` | Bot CrashBot (284 paires USDC, dip-buy) | — |
| `tradex-momentum` | Bot Momentum (7 paires USD, Revolut X) | — |
| `tradex-dashboard-unified` | Dashboard Streamlit unifié | 8502 |

### Commandes utiles

```bash
# Logs en direct
ssh BOT-VPS 'sudo journalctl -u tradex-binance -f'
ssh BOT-VPS 'sudo journalctl -u tradex-binance-crashbot -f'
ssh BOT-VPS 'sudo journalctl -u tradex-momentum -f'

# État de tous les services
ssh BOT-VPS 'for svc in tradex-binance tradex-binance-crashbot tradex-momentum tradex-dashboard-unified; do echo -n "$svc: "; sudo systemctl is-active $svc; done'

# Redémarrer un bot
ssh BOT-VPS 'sudo systemctl restart tradex-binance'

# Dashboard
# http://213.199.41.168:8502
```
