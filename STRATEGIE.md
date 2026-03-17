# 🧠 TradeX — Comment ça marche (version simple)

## Table des matières

1. [L&#39;idée générale](#lidée-générale)
2. [Les cinq bots](#les-cinq-bots)
3. [Bot 1 — Trail Range (Binance)](#bot-1--trail-range-binance)
   - [Détecter le range](#détecter-le-range)
   - [Décider quand entrer](#décider-quand-entrer-range)
   - [OCO : TP + SL en un seul ordre](#oco--tp--sl-en-un-seul-ordre)
   - [Trail@TP : le trailing OCO](#trailtp--le-trailing-oco)
   - [Sortie forcée si tendance confirmée](#sortie-forcée-si-tendance-confirmée)
4. [Bot 2 — CrashBot (Binance)](#bot-2--crashbot-binance)
   - [L&#39;idée](#lidée-crashbot)
   - [Détecter un crash](#détecter-un-crash)
   - [Entrée et Stop Loss](#entrée-et-stop-loss-crashbot)
   - [Step Trailing : le TP qui monte par paliers](#step-trailing--le-tp-qui-monte-par-paliers)
   - [Kill-Switch mensuel](#kill-switch-mensuel)
5. [Bot 3 — Listing Bot (Binance)](#bot-3--listing-bot-binance)
   - [L&#39;idée](#lidée-listing)
   - [Détection de listing](#détection-de-listing)
   - [Filtre Momentum](#filtre-momentum)
   - [OCO dynamique en 2 phases](#oco-dynamique-en-2-phases)
   - [Force close](#force-close)
6. [Bot 4 — London Breakout (Revolut X)](#bot-4--london-breakout-revolut-x)
   - [L&#39;idée](#lidée-london)
   - [Phase 1 : Accumuler le range de session](#phase-1--accumuler-le-range-de-session)
   - [Phase 2 : Détecter le breakout](#phase-2--détecter-le-breakout)
   - [Phase 3 : Entrée et gestion](#phase-3--entrée-et-gestion)
7. [Exécution des ordres Revolut X](#exécution-des-ordres-revolut-x-london--infinity)
8. [Bot 5 — Infinity Bot (Revolut X)](#bot-5--infinity-bot-revolut-x)
   - [L&#39;idée](#lidée-infinity)
   - [Trailing High](#trailing-high--le-prix-de-référence-dynamique)
   - [Paliers d&#39;achat](#paliers-dachat-dca-inversé)
   - [Paliers de vente](#paliers-de-vente-distribution-progressive)
   - [Sécurités](#sécurités)
9. [Bot 6 — DCA RSI v2 (Revolut X)](#bot-6--dca-rsi-v2-revolut-x)
   - [L&#39;idée](#lidée-dca)
   - [Brackets RSI + MVRV progressif](#brackets-rsi--mvrv-progressif)
   - [Market Regime (MA200)](#market-regime-ma200)
   - [Spending Caps](#spending-caps)
   - [Crash Reserve](#crash-reserve)
   - [Budget et horizon](#budget-et-horizon)
10. [Allocation dynamique du capital](#allocation-dynamique-du-capital)
   - [Comment ça marche](#comment-ça-marche)
   - [Pourquoi cette logique](#pourquoi-cette-logique)
11. [Gestion du risque (Money Management)](#gestion-du-risque-money-management)
12. [La boucle de chaque bot](#la-boucle-de-chaque-bot)
13. [Les fichiers et qui fait quoi](#les-fichiers-et-qui-fait-quoi)
14. [Exemple concret — Trade RANGE](#exemple-concret--trade-range)
15. [Exemple concret — Trade CRASH](#exemple-concret--trade-crash)
16. [Exemple concret — Trade LISTING](#exemple-concret--trade-listing)
17. [Exemple concret — Trade LONDON BREAKOUT](#exemple-concret--trade-london-breakout)
18. [Exemple concret — Trade INFINITY](#exemple-concret--trade-infinity)
19. [Exemple concret — Trade DCA RSI](#exemple-concret--trade-dca-rsi)
20. [Ce que les bots ne font PAS](#ce-que-les-bots-ne-font-pas)
21. [Les paramètres importants](#les-paramètres-importants-fichier-env)
22. [Infrastructure &amp; Déploiement](#infrastructure--déploiement)

---

## L'idée générale

TradeX est un **écosystème de 6 bots** qui surveillent le marché crypto **24h/24** et tradent automatiquement quand les conditions sont réunies.

Les bots fonctionnent sur **2 exchanges** avec **6 stratégies complémentaires** :

> **🔄 Trail Range** (Binance) : "Quand le prix oscille dans un couloir, je joue les rebonds entre le plafond et le plancher."
> **💥 CrashBot** (Binance) : "Quand une crypto s'effondre brutalement, j'achète le dip et je laisse remonter."
> **🆕 Listing Bot** (Binance) : "Quand un nouveau token est listé sur Binance, j'achète le momentum initial et je sécurise via OCO dynamique."
> **🇬🇧 London Breakout** (Revolut X) : "Quand le prix casse le range de la session de Londres (08-16 UTC), j'entre long avec un SL basé sur l'ATR."
> **♾️ Infinity** (Revolut X) : "Quand une crypto baisse de X% par rapport à son plus haut récent, j'accumule par paliers DCA puis je revends progressivement."
> **📈 DCA RSI v2** (Revolut X) : "Chaque jour, j'achète BTC et ETH selon le RSI, le régime de marché (MA200) et le MVRV on-chain (progressif), avec des spending caps et une réserve crash."

Aucun bot ne prédit l'avenir. Chacun **constate** un pattern spécifique et agit en conséquence.

---

## Les six bots

| Bot                     | Exchange        | Paires                                 | Logique                              | Side                | Capital                     |
| ----------------------- | --------------- | -------------------------------------- | ------------------------------------ | ------------------- | --------------------------- |
| 🔄**Trail Range** | Binance (USDC)  | ~284 (auto-discovery)                  | Rebonds dans le range + trailing OCO | Long Only           | 5–20% Binance (dynamique) |
| 💥**CrashBot**    | Binance (USDC)  | ~284 (auto-discovery)                  | Dip-buy + step-trail                 | **Long Only** | 50–65% Binance (dynamique) |
| 🆕**Listing Bot** | Binance (USDC)  | Auto-discovery nouveaux USDC           | Listing event + momentum + OCO       | **Long Only** | 30% Binance (fixe) |
| 🇬🇧**London Breakout** | Revolut X (USD) | 8 (BTC, ETH, SOL, BNB, LINK, ADA, DOT, AVAX) | Session breakout (08-16 UTC) → long | **Long Only** | 20% Revolut X |
| ♾️**Infinity**  | Revolut X (USD) | BTC, AAVE, XLM (configs optimisées) | DCA inversé + vente paliers         | **Long Only** | 65% Revolut X (~22% par paire) |
| 📈**DCA RSI v2** | Revolut X (USD) | BTC, ETH                               | DCA quotidien RSI + regime MA200 + MVRV progressif + caps + crash reserve | **Long Only** | DCA_CAPITAL_PCT du solde RevX (85/15) |

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

| Signal         | Condition                        | Logique                                        |
| -------------- | -------------------------------- | ---------------------------------------------- |
| **BUY**  | Prix ≤ Range Low × (1 + 0.2%)  | "Le prix touche le plancher, il va remonter"   |
| **SELL** | Prix ≥ Range High × (1 - 0.2%) | "Le prix touche le plafond, il va redescendre" |

C'est de la **mean-reversion** : on parie que le prix revient au centre du range.

### OCO : TP + SL en un seul ordre

📄 **Fichier : `bot_binance.py`**

Binance supporte les **ordres OCO** (One-Cancels-Other) : un Take Profit ET un Stop Loss sont posés simultanément. Quand l'un est touché, l'autre est automatiquement annulé.

|              | Valeur                                | Logique                         |
| ------------ | ------------------------------------- | ------------------------------- |
| **TP** | Milieu du range                       | "Le prix revient au centre"     |
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

| Paramètre         | Valeur | Description                     |
| ------------------ | ------ | ------------------------------- |
| `drop_threshold` | 20%    | Baisse minimum pour déclencher |
| `lookback_bars`  | 12     | 12 × 4h = 48 heures de recul   |

### Entrée et Stop Loss (CrashBot)

|                      | Formule                     | Logique                              |
| -------------------- | --------------------------- | ------------------------------------ |
| **Entrée**    | Market order au prix actuel | On achète immédiatement le dip     |
| **TP initial** | Entrée × (1 + 8%)         | Objectif : +8% de rebond             |
| **SL**         | Entrée - 1.5 × ATR        | Protection basée sur la volatilité |

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

### Momentum Sizing (taille dynamique)

Le risk% par trade n'est pas fixe : il **augmente après un gain** et **diminue après une perte**.

**Pourquoi ?** L'analyse de 1018 trades historiques a révélé que les résultats du CrashBot ne sont **pas aléatoires** : ils forment des **séries** (autocorrélation +0.24 à +0.55). Après un gain, la probabilité d'un autre gain est élevée → on augmente l'exposition. Après une perte, le risque de série perdante est fort → on réduit.

| Événement | Action | Formule |
|-----------|--------|---------|
| **WIN** (PnL ≥ 0) | Boost | `risk = min(risk × 1.2, 10%)` |
| **LOSS** (PnL < 0) | Shrink | `risk = max(risk × 0.8, 2%)` |

```
Exemple :
  risk de base : 5%
  Trade 1 → WIN  → risk = 5% × 1.2 = 6.0%
  Trade 2 → WIN  → risk = 6.0% × 1.2 = 7.2%
  Trade 3 → LOSS → risk = 7.2% × 0.8 = 5.76%
  Trade 4 → LOSS → risk = 5.76% × 0.8 = 4.6%
  Trade 5 → LOSS → risk = 4.6% × 0.8 = 3.7%
  ...plancher à 2%, plafond à 10%
```

| Paramètre | Valeur | Variable d'environnement |
|-----------|--------|--------------------------|
| Activation | `true` | `BINANCE_CRASHBOT_MOMENTUM_SIZING` |
| Boost (après WIN) | ×1.2 | `BINANCE_CRASHBOT_RISK_BOOST_MULT` |
| Shrink (après LOSS) | ×0.8 | `BINANCE_CRASHBOT_RISK_SHRINK_MULT` |
| Plancher | 2% | `BINANCE_CRASHBOT_MIN_RISK_PCT` |
| Plafond | 10% | `BINANCE_CRASHBOT_MAX_RISK_PCT` |

Le risk% actuel est persisté sur disque (`crashbot_momentum_state.json`) et restauré au redémarrage.

---

## Bot 3 — Listing Bot (Binance)

### L'idée (Listing)

> Quand Binance ajoute un nouveau token (paire USDC), le prix explose souvent dans les premières minutes. Le bot détecte le listing en temps réel, vérifie un filtre momentum, achète immédiatement, et sécurise via un OCO dynamique en 2 phases.

**⚠️ LONG ONLY** : le bot n'achète que les nouveaux listings. Pas de short.

### Détection de listing

📄 **Fichier : `listing_detector.py` + `bot_binance_listing.py`**

Le bot poll l'API Binance `exchangeInfo` toutes les 30 secondes. Un diff avec les symboles connus (`listing_known_symbols.json`) détecte les nouvelles paires USDC en status `TRADING` :

```
Symboles connus : 284 paires USDC

Nouveau poll exchangeInfo → 285 paires
Diff : NEWUSDC ajouté !
→ 🔔 Nouveau listing détecté : NEWUSDC
```

### Filtre Momentum

Une fois un nouveau symbol détecté, le bot récupère les **klines 1 minute** et vérifie si le HIGH de la première bougie atteint au moins **+30%** au-dessus de l'OPEN.

```
Kline 1m du listing :
  OPEN = 1.00$ | HIGH = 1.42$
  Momentum = (1.42 - 1.00) / 1.00 = +42% ≥ 30% ✅
→ 🆕 Momentum validé ! Signal d'achat.
```

Ce filtre élimine ~52% des listings de mauvaise qualité (les "dead listings" qui ne bougent pas).

### Entrée

|                      | Valeur                          | Logique                              |
| -------------------- | ------------------------------- | ------------------------------------ |
| **Entrée**     | Market order immédiat           | Capturer le momentum initial         |
| **Capital**    | 30% Binance / max_slots         | Max 3 positions simultanées          |
| **Plafond**    | 5 000 USDC par slot             | Protection contre la surexposition   |

### OCO dynamique en 2 phases

Une fois entré, le bot place un **OCO1** (SL + TP simultanés). Si le prix approche du TP, l'OCO est **re-armé** avec de nouveaux niveaux.

**Phase 1 — OCO1 (initial) :**

| Niveau       | Formule                    | Exemple (entrée = 1.00$) |
| ------------ | -------------------------- | ------------------------ |
| **SL**  | Entry × 0.92 (-8%)        | 0.92$                    |
| **TP1** | Entry × 1.30 (+30%)       | 1.30$                    |

**Phase 2 — Re-arm OCO2 :**

Quand le prix atteint ≥ TP1 × 0.98 (= 98% du TP1), le bot annule OCO1 et place OCO2 :

| Niveau       | Formule                    | Exemple (TP1 = 1.30$)   |
| ------------ | -------------------------- | ------------------------ |
| **SL2** | TP1 × 0.769 (≈ breakeven) | 1.00$ (≈ entry)         |
| **TP2** | TP1 × 1.538 (+100%)       | 2.00$                    |

```
Entrée : 1.00$ | SL = 0.92$ | TP1 = 1.30$

  Prix monte à 1.28$ (≥ 1.30 × 0.98 = 1.274) → Re-arm !
    Annule OCO1
    Nouveau SL2 = 1.30 × 0.769 = 1.00$ (≈ breakeven)
    Nouveau TP2 = 1.30 × 1.538 = 2.00$ (+100%)

  Prix monte à 2.00$ → TP2 touché !
  Gain : (2.00 - 1.00) × size = gros profit 🎉
```

### Force close

Si un trade est ouvert depuis plus de **7 jours** sans toucher TP ni SL → vente market forcée. Cela évite de bloquer du capital sur un listing qui ne bouge plus.

### Résultats backtest (2 ans, 288 paires)

```
Config : SL -8%, TP +30%, re-arm (SL2=0.769×TP1, TP2=1.538×TP1)
Momentum ≥ 30%, Horizon 7j, 3 slots max

Scénario réaliste :  +1,571%  (fill rate 50%, slippage 1%)
Scénario optimiste : +45,608% (fill rate 100%, pas de slippage)
```

---

## Bot 4 — London Breakout (Revolut X)

### L'idée (London)

> La session de Londres (08h–16h UTC) génère souvent un range consolidé. Quand le prix casse au-dessus de ce range après 16h UTC, avec du volume et un range suffisamment large, le bot entre long.

C'est un signal en **3 phases** :

### Phase 1 : Accumuler le range de session

📄 **Fichier : `bot_london.py`**

Le bot accumule le **high/low** des bougies H4 entre 08:00 et 16:00 UTC (bougies 08:00 et 12:00) :

```
Bougie H4 08:00 : High = 3 520$ | Low = 3 480$
Bougie H4 12:00 : High = 3 540$ | Low = 3 475$

→ Session range : High = 3 540$ | Low = 3 475$
→ Largeur = (3540 - 3475) / 3475 = 1.87% ≥ 1.5% ✅
```

### Phase 2 : Détecter le breakout

Après 16:00 UTC, le bot vérifie si le prix a cassé le haut du range avec **3 filtres** :

| # | Critère                    | Condition                  | Pourquoi                                      |
| - | --------------------------- | -------------------------- | --------------------------------------------- |
| 1 | **Breakout haussier** | Close > session_high       | Le prix a cassé au-dessus du range           |
| 2 | **Volume explosif**   | Volume ≥ 2.0× MA20       | Confirmation par la liquidité                |
| 3 | **Range suffisant**   | Range ≥ 1.5% du prix     | Éviter les micro-ranges sans potentiel       |

```
Bougie H4 16:00 :
  Close: 3 560$ > session_high 3 540$ ✅
  Volume: 12M (moy 20: 5M → 2.4×) ✅
  Range: 1.87% ≥ 1.5% ✅
  → 🇬🇧 BREAKOUT HAUSSIER détecté !
```

### Phase 3 : Entrée et gestion

|                   | Valeur                                | Logique                        |
| ----------------- | ------------------------------------- | ------------------------------ |
| **Entrée** | Ordre limit maker (0% frais)          | Revolut X maker = 0% fee       |
| **SL**      | Entry - 2.0 × ATR(14)               | Basé sur la volatilité récente |
| **TP1**     | Entry × 1.02 (+2%) → vend 50%       | Sécuriser rapidement          |
| **TP2**     | Entry × 1.05 (+5%) → vend le reste  | Laisser courir le profit       |
| **Breakeven** | SL ramené à l'entrée après TP1   | Protection du capital          |
| **Risque**  | 5% du capital (`LON_RISK_PERCENT`)   | Par trade                      |
| **Cooldown** | 2 bougies H4 (8h) entre trades     | Éviter le surtrading           |

---

## Exécution des ordres Revolut X (London + Infinity + DCA RSI)

📄 **Fichier : `revolut_client.py`**

Revolut X facture **0% en maker** et 0.09% en taker. Les deux bots Revolut X (London et Infinity) utilisent le même mécanisme d'exécution.

### Infinity Bot (BUY) — retry maker 2× + taker fallback

```
Maker #1 (prix initial)     → attente 60s → fill? ✅ OK (0% fee)
       ↓ no-fill
Rafraîchir le prix
Maker #2 (prix actualisé)   → attente 60s → fill? ✅ OK (0% fee)
       ↓ no-fill
Rafraîchir le prix
Taker fallback              → fill immédiat ✅ (0.09% fee)
```

### Infinity Bot (SELL) — maker + taker fallback

1 tentative maker, puis taker fallback immédiat (ne pas rater la sortie).

### London Bot — maker + taker fallback

1 tentative maker (60s via `LON_MAKER_WAIT_SECONDS`), puis fallback en taker (0.09%).

### DCA RSI Bot — maker + taker fallback

1 tentative maker (60s via `DCA_MAKER_WAIT_SECONDS`), puis fallback en taker (0.09%). Même logique que London.

---

## Bot 5 — Infinity Bot (Revolut X)

C'est le bot **DCA inversé** multi-paires. Il achète les baisses par paliers et revend progressivement quand le prix remonte.

### L'idée

> "Quand une crypto baisse de X% par rapport à son plus haut récent, j'achète. S'il continue de baisser, j'achète encore à des paliers plus profonds. Puis je revends par paliers quand il remonte."

C'est l'opposé d'un grid bot : au lieu d'acheter et vendre sur des niveaux fixes, le prix de référence suit dynamiquement le marché (trailing high).

### Paires tradées et configs optimisées

Chaque paire a une config **walk-forward validée** (entraînement 2020→2024, test 2024→2026) :

| Paire    | Trail | Entry Drop | Buy Levels           | Sell Levels                    | SL   | Test PnL   | Test PF |
| -------- | ----- | ---------- | -------------------- | ------------------------------ | ---- | ---------- | ------- |
| BTC-USD  | 72 bars (12j) | -5% | -5%,-10%,-15%,-20%,-25% | +0.8%,+1.5%,+2.2%,+3%,+4%  | 15%  | (en prod)  | —      |
| AAVE-USD | 48 bars (8j)  | -12% | -12%,-20%,-28%,-35%,-42% | +2%,+4%,+6%,+8%,+12%       | 25%  | **+47.20%** | **4.92** |
| XLM-USD  | 48 bars (8j)  | -12% | -12%,-20%,-28%,-35%,-42% | +0.8%,+1.5%,+2.2%,+3%,+4% | 25%  | **+26.13%** | **32.66** |

> **Pourquoi ces 3 paires ?** Un grid search de 336 combinaisons par paire + validation out-of-sample a identifié AAVE et XLM comme les meilleurs candidats. ETH et SOL ont été rejetés (surfit en test).

### Trailing High — le prix de référence dynamique

📄 **Fichier : `infinity_engine.py`**

Le bot calcule en permanence le **plus haut des N dernières bougies H4** (configurable par paire : 72 bars pour BTC, 48 bars pour AAVE/XLM) :

```
│    ╱★ Trailing High (plus haut sur N bars H4)
│   ╱  ╲
│  ╱    ╲
│ ╱      ╲──── Prix actuel
│╱         ╲
│            ╲── Drop ≥ seuil → Premier achat !
```

Quand le prix **chute de ≥ seuil** (5% pour BTC, 12% pour AAVE/XLM) par rapport à ce trailing high → le bot déclenche le premier achat.

### Paliers d'achat (DCA inversé) — BTC

| Palier | Drop depuis la référence | % du capital     | RSI gate                        |
| ------ | -------------------------- | ---------------- | ------------------------------- |
| L1     | -5%                        | 25%              | RSI < 50 → full, 30-50 → demi |
| L2     | -10%                       | 20%              | idem                            |
| L3     | -15%                       | 15%              | idem                            |
| L4     | -20%                       | 10%              | idem                            |
| L5     | -25%                       | Reste disponible | idem                            |

### Paliers d'achat — AAVE & XLM

| Palier | Drop depuis la référence | % du capital     | RSI gate                        |
| ------ | -------------------------- | ---------------- | ------------------------------- |
| L1     | -12%                       | 25%              | RSI < 50 → full, 30-50 → demi |
| L2     | -20%                       | 20%              | idem                            |
| L3     | -28%                       | 15%              | idem                            |
| L4     | -35%                       | 10%              | idem                            |
| L5     | -42%                       | Reste disponible | idem                            |

> Les altcoins ont des paliers plus larges car ils sont plus volatils que BTC.

### Paliers de vente (distribution progressive)

Quand le prix remonte au-dessus du **PMP (Prix Moyen Pondéré)**, le bot vend 20% à chaque palier.

**BTC :**
| Palier | Distance du PMP | Action                              |
| ------ | --------------- | ----------------------------------- |
| TP1    | +0.8%           | Vend 20% + active le breakeven stop |
| TP2    | +1.5%           | Vend 20%                            |
| TP3    | +2.2%           | Vend 20%                            |
| TP4    | +3.0%           | Vend 20%                            |
| TP5    | +4.0%           | Vend tout le reste                  |

**AAVE :**
| Palier | Distance du PMP | Action                              |
| ------ | --------------- | ----------------------------------- |
| TP1    | +2.0%           | Vend 20% + active le breakeven stop |
| TP2    | +4.0%           | Vend 20%                            |
| TP3    | +6.0%           | Vend 20%                            |
| TP4    | +8.0%           | Vend 20%                            |
| TP5    | +12.0%          | Vend tout le reste                  |

**XLM :** Mêmes paliers de vente que BTC.

### Sécurités

- **Breakeven stop** : Après TP1, si le prix retombe au PMP → vente totale (pas de perte)
- **Stop-loss** : Si le prix baisse sous le PMP → vente market (BTC: -15%, AAVE/XLM: -25%)
- **Override sell** : Si le prix monte de +20% au-dessus du PMP → vente totale immédiate
- **Max investi** : 70% du capital alloué maximum par cycle
- **Pas de RSI gate sur les ventes** (clé de performance — +107% vs +50% avec)

### Capital et allocation

Le capital Revolut X est partagé entre **Infinity** (3 paires) et **London Breakout** :

| Bot           | % du capital Revolut X |
| ------------- | ---------------------- |
| ♾️ Infinity | **80%** (≈27% par paire × 3) |
| 🇬🇧 London Breakout | **20%** |

### Résultats backtest & walk-forward (6 ans, 2020-2026)

**BTC-USD (config par défaut) :**
```
Rendement : +107.56%
Sharpe    : 3.85
PF        : 3.10
Max DD    : -15.78%
Win Rate  : 90% (18/20 trades gagnants)
Stops     : seulement 2 en 6 ans
```

**AAVE-USD (walk-forward test period 2024-2026) :**
```
Rendement : +47.20%
PF        : 4.92
Max DD    : -7.90%
Stops     : 1
```

**XLM-USD (walk-forward test period 2024-2026) :**
```
Rendement : +26.13%
PF        : 32.66
Max DD    : -4.54%
Stops     : 0
```

---

## Bot 6 — DCA RSI v2 (Revolut X)

📄 **Fichiers : `dca_engine.py` (logique pure), `bot_dca.py` (boucle)**

### L'idée (DCA)

Au lieu de timer le marché, le bot **achète BTC et ETH chaque jour** pour un montant qui dépend de 3 facteurs :
1. **RSI quotidien BTC** → brackets d'achat (montant de base)
2. **MVRV on-chain** → multiplicateur progressif (×1.5 si < 1.0, ×2.0 si < 0.85)
3. **Régime de marché** (MA200) → allocation BTC/ETH dynamique

En cas de crash majeur, une **réserve de crash** (15% du capital) permet des achats bonus. Des **spending caps** ($1,500/mois, $400/semaine) protègent contre l'épuisement trop rapide du budget.

L'objectif est d'**accumuler progressivement** sur 12+ mois, en achetant plus quand le marché est en solde.

### Brackets RSI + MVRV progressif

Le RSI quotidien de BTC détermine le **montant de base** du jour :

```
RSI > 70 (Overbought)   →   0$ (on n'achète pas, trop cher)
55 < RSI ≤ 70 (Warm)    →  30$ (achat léger)
45 ≤ RSI ≤ 55 (Neutral) →  60$ (achat normal)
RSI < 45 (Oversold)     →  90$ (achat agressif)
```

Le **MVRV** (Market Value to Realized Value) applique ensuite un **multiplicateur progressif** :

```
MVRV ≥ 1.0              → ×1.0 (pas de boost)
0.85 ≤ MVRV < 1.0       → ×1.5 (sous-évaluation modérée)
MVRV < 0.85              → ×2.0 (sous-évaluation profonde)
```

Le montant final est plafonné à **$150/jour** (`DCA_MAX_DAILY_BUY`). Un **boost cooldown** de 24h empêche les boosts MVRV trop rapprochés (seuil : montant > $120).

Le MVRV est récupéré via **CoinMetrics Community API** (gratuit, sans clé API), caché 1h.

```
Exemple : RSI = 48, MVRV = 0.92
  Bracket NEUTRAL → 60$ (base)
  MVRV < 1.0 → ×1.5 = 90$ (final)
  Split regime NORMAL : BTC 90% = 81.00$ | ETH 10% = 9.00$
```

### Market Regime (MA200)

Le prix BTC par rapport à la **SMA 200 jours** détermine le régime de marché :

| Régime | Condition | Allocation | Description |
|--------|-----------|------------|-------------|
| **NORMAL** | prix > MA200 | BTC 90% / ETH 10% | Marché haussier standard |
| **WEAK** | prix < MA200 | BTC 95% / ETH 5% | Marché fragilisé, on concentre sur BTC |
| **CAPITULATION** | prix < MA200 × 0.85 | BTC 100% / ETH 0% | Bear market profond, full BTC |

Le régime est recalculé à chaque `_initialize()` et affiché dans le heartbeat.

### Spending Caps

Pour éviter d'épuiser le budget trop vite (surtout en bear market avec des boosts MVRV fréquents) :

| Cap | Montant | Reset |
|-----|---------|-------|
| **Mensuel** | $1,500 | 1er de chaque mois |
| **Hebdomadaire** | $400 | Chaque lundi (ISO week) |

Si un achat dépasse le cap restant, le montant est automatiquement réduit au montant disponible.

### Crash Reserve

Une réserve de **15% du capital DCA** est dédiée aux crashes majeurs de BTC. L'ancre de crash est calculée comme le **max du rolling high 90j et du rolling high 180j** pour éviter les faux signals :

| Drop BTC vs Crash Anchor | Montant bonus (% de réserve) | Exemple (reserve $1,967) |
| ------------------------- | ----------------------------- | ------------------------ |
| -15%                      | 25% de la réserve (BTC only) | ~$492 |
| -25%                      | 35% de la réserve (BTC only) | ~$688 |
| -35%                      | 40% de la réserve (BTC only) | ~$787 |

- Chaque niveau ne se déclenche qu'**une seule fois** par crash
- Quand le prix remonte au-dessus de -10% du high → les niveaux se **réinitialisent**

### Budget et horizon

```
Capital DCA      : solde Revolut X × DCA_CAPITAL_PCT (défaut 100%)
├─ DCA actif     : 85% du capital DCA (achats quotidiens)
└─ Crash reserve : 15% du capital DCA (achats bonus crash)

  Budget calculé dynamiquement au démarrage du bot.
  Rythme max : 150$/jour (plafonné)
  Rythme moyen : ~30-90$/jour

Protections :
  Monthly cap   : $1,500/mois
  Weekly cap    : $400/semaine
  Boost cooldown: 24h entre deux boosts MVRV

Signaux on-chain : MVRV via CoinMetrics Community API (gratuit)
Exécution : 1×/jour à 10:00 UTC, maker-only (0% frais Revolut X)
```

Le bot s'arrête automatiquement quand le budget (DCA actif ou crash reserve) est épuisé.

### Observabilité v2 — DCADecision

Chaque décision d'achat est loggée dans Firebase (`events` collection, type `DCA_DECISION`) avec un objet `DCADecision` contenant :
- RSI, bracket, montant de base et montant final
- MVRV, multiplicateur MVRV, cap appliqué (oui/non)
- Régime de marché, allocation BTC/ETH
- Raison textuelle (ex: "NEUTRAL RSI=48.2, MVRV=0.92 → ×1.5, regime=NORMAL")

Ces données alimentent le **dashboard Analytics** (onglet DCA) : cockpit régime, graphe MVRV, distribution brackets, log des décisions.

---

## Allocation dynamique du capital

📄 **Fichier : `allocator.py`**

### Comment ça marche

Les bots **Trail Range**, **CrashBot** et **Listing Bot** partagent le même capital sur Binance. La répartition est recalculée **1×/jour** automatiquement.

Le **Listing Bot reçoit toujours 30%** du capital (fixe). Les **70% restants** sont répartis entre Trail Range et CrashBot selon le **Profit Factor (PF)** du Trail Range sur 90 jours :

```
PF = somme des gains / |somme des pertes|

PF > 1 → le bot gagne plus qu'il ne perd
PF < 1 → le bot perd plus qu'il ne gagne
```

| PF Trail Range (90j)    | Trail Range   | CrashBot      | Listing Bot   | Régime        |
| ----------------------- | ------------- | ------------- | ------------- | -------------- |
| PF < 0.9 OU < 20 trades | **5%** | **65%** | **30%** | 🛡️ DÉFENSIF |
| 0.9 ≤ PF ≤ 1.1        | **10%** | **60%** | **30%** | ⚖️ NEUTRE    |
| PF > 1.1                | **20%** | **50%** | **30%** | 🚀 AGRESSIF    |

```
Exemple : Capital Binance = 3 226 USDC, PF Trail = 0.47

  PF 0.47 < 0.9 → Régime DÉFENSIF
  Trail Range : 5%  = 161 USDC
  CrashBot    : 65% = 2 097 USDC
  Listing Bot : 30% = 968 USDC
```

### Pourquoi cette logique

- **CrashBot domine par défaut** car sa stratégie (dip-buy) est plus robuste en marché latéral
- **Listing Bot a une part fixe de 30%** car les opportunités de listing sont indépendantes du PF
- Quand le Trail Range **prouve** qu'il gagne (PF > 1.1), on lui donne plus de capital
- Quand il perd, on réduit son exposition et CrashBot prend le relais

### Les bots Revolut X (London + Infinity + DCA RSI)

Les bots London et Infinity partagent le capital Revolut X avec une allocation fixe : **80% Infinity** / **20% London Breakout**. Le **DCA RSI v2** opère avec un **budget dynamique** (`DCA_CAPITAL_PCT` du solde Revolut X, réparti 85% actif / 15% crash reserve) et des spending caps ($1,500/mois, $400/semaine). Les deux exchanges sont complètement séparés.

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

| Bot            | Exchange  | Risque/trade | Max positions | Max % capital/position |
| -------------- | --------- | ------------ | ------------- | ---------------------- |
| 🔄 Trail Range | Binance   | 5%           | 3             | 30%                    |
| 💥 CrashBot    | Binance   | 2%           | 3             | 30%                    |
| � Listing Bot | Binance   | N/A (market) | 3             | 30% / slots            |
| �🇬🇧 London Breakout | Revolut X | 5%           | 1             | 50%                    |
| ♾️ Infinity  | Revolut X | 15-25% (SL)  | 3 cycles max  | 70% (max investi/paire) |
| 📈 DCA RSI   | Revolut X | N/A (DCA)    | N/A            | DCA_CAPITAL_PCT du solde |

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

### � Listing Bot (`bot_binance_listing.py`)

```
┌──────────────────────────────────────────────────────────┐
│ TOUTES LES 10 SECONDES                                  │
│                                                          │
│  1. Poll exchangeInfo (cache 30s)                        │
│     ├─ Nouveau symbole USDC ? → 🔔 Détecté !            │
│     │   ├─ Récupérer klines 1m                           │
│     │   ├─ Momentum ≥ 30% ? → Signal valide             │
│     │   │   └─ Market BUY + OCO1 (SL -8%, TP +30%)      │
│     │   └─ Momentum < 30% → ⏭️ Skip (dead listing)     │
│     └─ Pas de nouveau symbole → continuer                │
│                                                          │
│  2. Gérer les positions ouvertes                         │
│     ├─ Prix ≥ TP1 × 0.98 ? → Re-arm OCO2                │
│     │   └─ SL2 = TP1×0.769 | TP2 = TP1×1.538            │
│     ├─ OCO fill SL/TP ? → Clôturer position              │
│     └─ Ouverte > 7 jours ? → ⏰ Force close market      │
│                                                          │
│ 1×/JOUR                                                  │
│    └─ Recalculer l'allocation dynamique (30% fixe)       │
│                                                          │
│ TOUTES LES 10 MINUTES                                    │
│    └─ 💓 Heartbeat Telegram + Firebase                   │
└──────────────────────────────────────────────────────────┘
```

### �🇬🇧 London Breakout (`bot_london.py`)

```
┌──────────────────────────────────────────────────────────┐
│ TOUTES LES 30 SECONDES                                  │
│                                                          │
│  Pour chaque paire (8) :                                 │
│    ├─ 08:00-16:00 UTC ? → Accumuler le range de session  │
│    │   └─ Mise à jour session_high / session_low          │
│    ├─ Après 16:00 UTC ? → Vérifier breakout              │
│    │   ├─ Close > session_high ? ✅                       │
│    │   ├─ Volume ≥ 2× MA20 ? ✅                          │
│    │   ├─ Range ≥ 1.5% ? ✅                              │
│    │   └─ Tout OK → 🇬🇧 SIGNAL LONDON BREAKOUT          │
│    │       └─ Placer ordre limit maker (0% frais)         │
│    └─ Position ouverte ? → Gérer TP1/TP2/BE/SL           │
│        ├─ TP1 (+2%) → vend 50% + breakeven SL            │
│        └─ TP2 (+5%) → vend le reste                      │
│                                                          │
│ TOUTES LES 10 MINUTES                                    │
│    └─ Heartbeat Firebase + calcul equity                 │
│                                                          │
│ COOLDOWN : 8h (2 bougies H4) entre trades/paire          │
└──────────────────────────────────────────────────────────┘
```

### ♾️ Infinity (`bot_infinity.py`)

```
┌──────────────────────────────────────────────────────────┐
│ TOUTES LES 30 SECONDES (tick)                            │
│                                                          │
│  POUR CHAQUE PAIRE (BTC-USD, AAVE-USD, XLM-USD) :       │
│  1. Récupérer prix (ticker Revolut X)                    │
│  2. Selon la phase :                                     │
│                                                          │
│  WAITING (pas de position)                               │
│    ├─ Nouvelle bougie H4 ? → Évaluer conditions d'entrée │
│    │   ├─ Drop ≥ seuil vs trailing high ? ✅/❌          │
│    │   ├─ RSI(14) ≤ 50 ?                ✅/❌             │
│    │   └─ Tout OK → Premier achat L1 (25% du capital)     │
│    │       └─ Maker ×2 (prix rafraîchi) → taker fallback  │
│    └─ Pas de nouvelle H4 → Attendre                      │
│                                                          │
│  BUYING (accumulation DCA)                               │
│    ├─ Prix ≤ prochain palier ?                            │
│    │   └─ OUI → Achat palier suivant (L2..L5)            │
│    ├─ Breakeven stop actif + prix ≤ PMP ?                │
│    │   └─ OUI → Vente totale market (protection)         │
│    ├─ SL touché (PMP - 15%) ?                            │
│    │   └─ OUI → Vente totale market (stop loss)          │
│    └─ Prix ≥ TP1 ? → Passer en SELLING                   │
│                                                          │
│  SELLING (distribution progressive)                      │
│    ├─ Prix ≥ prochain TP ? → Vendre 20% au palier        │
│    │   ├─ TP1 (+0.8%) → vend 20% + active breakeven      │
│    │   ├─ TP2 (+1.5%) → vend 20%                         │
│    │   ├─ TP3 (+2.2%) → vend 20%                         │
│    │   ├─ TP4 (+3.0%) → vend 20%                         │
│    │   └─ TP5 (+4.0%) → vend tout le reste               │
│    ├─ Override : prix ≥ PMP + 20% → vente totale         │
│    ├─ Breakeven : prix ≤ PMP → vente totale              │
│    └─ Tout vendu → Cycle terminé → retour WAITING        │
│                                                          │
│ TOUTES LES 4 HEURES (nouvelle bougie H4)                 │
│    └─ Recalculer trailing high (72 bars = 12 jours)      │
│                                                          │
│ TOUTES LES 10 MINUTES                                    │
│    └─ Heartbeat : écart prix/cible, countdown H4,        │
│       dernière évaluation (drop/RSI), Firebase + Telegram │
│                                                          │
│ 1×/JOUR                                                  │
│    └─ Cleanup events Firebase (> 2 jours)                │
└──────────────────────────────────────────────────────────┘
```

### 📈 DCA RSI v2 (`bot_dca.py`)

```
┌──────────────────────────────────────────────────────────┐
│ TOUTES LES 60 SECONDES (tick)                            │
│                                                          │
│  0. Reset spending caps si nouveau mois/semaine          │
│     └─ monthly_spent → 0 si nouveau mois                │
│     └─ weekly_spent → 0 si nouvelle semaine ISO          │
│                                                          │
│  1. Vérifier la crash reserve (en permanence)            │
│     ├─ Crash anchor = max(high90j, high180j)             │
│     ├─ Drop -15% ? → Buy 25% reserve (BTC only)         │
│     ├─ Drop -25% ? → Buy 35% reserve (BTC only)         │
│     ├─ Drop -35% ? → Buy 40% reserve (BTC only)         │
│     └─ Remonté > -10% ? → Reset des niveaux crash       │
│                                                          │
│  2. DCA quotidien (1×/jour à 10:00 UTC)                  │
│     ├─ Déjà acheté aujourd'hui ? → Skip                 │
│     ├─ Budget DCA épuisé ? → Skip                       │
│     ├─ Classify RSI daily BTC → bracket + montant base   │
│     │   ├─ RSI > 70 → 0$ (overbought, skip)             │
│     │   ├─ 55 < RSI ≤ 70 → 30$ (warm)                   │
│     │   ├─ 45 ≤ RSI ≤ 55 → 60$ (neutral)                │
│     │   └─ RSI < 45 → 90$ (oversold)                     │
│     ├─ MVRV multiplier progressif                        │
│     │   ├─ MVRV ≥ 1.0 → ×1.0                            │
│     │   ├─ 0.85 ≤ MVRV < 1.0 → ×1.5                     │
│     │   └─ MVRV < 0.85 → ×2.0                            │
│     ├─ Cap journalier : min(montant, $150)               │
│     ├─ Check spending caps (monthly $1500, weekly $400)   │
│     ├─ Check boost cooldown (24h si montant > $120)      │
│     ├─ Regime allocation (MA200)                         │
│     │   ├─ NORMAL  → BTC 90% / ETH 10%                  │
│     │   ├─ WEAK    → BTC 95% / ETH 5%                   │
│     │   └─ CAPITUL → BTC 100% / ETH 0%                  │
│     ├─ Log DCADecision → Firebase (events)               │
│     └─ Placer ordres maker (0% frais)                    │
│                                                          │
│ TOUTES LES 10 MINUTES                                    │
│    └─ 💓 Heartbeat v2 : RSI, MVRV (×mult), MA200,       │
│       régime, budget, caps (mois/sem), Firebase+Telegram │
│                                                          │
│ BUDGET ÉPUISÉ → Bot s'arrête automatiquement           │
└──────────────────────────────────────────────────────────┘
```

---

## Les fichiers et qui fait quoi

| Fichier                     | Rôle en une phrase                                                     |
| --------------------------- | ----------------------------------------------------------------------- |
| `config.py`               | Charge les paramètres depuis `.env` (clés API, % de risque, etc.)   |
| `models.py`               | Définit les "objets" : bougie, swing, tendance, ordre, position, range |
| `swing_detector.py`       | Trouve les sommets et les creux dans les bougies                        |
| `trend_engine.py`         | Classifie la tendance : BULLISH, BEARISH ou NEUTRAL                     |
| `strategy_mean_rev.py`    | 🔄 Stratégie Mean Reversion : signaux BUY/SELL dans le range           |
| `strategy_trend.py`       | Stratégie Trend Following : signaux BUY/SELL en tendance               |
| `crashbot_detector.py`    | 💥 Détecte les crashes (drop ≥ 20%) + step trailing                   |
| `indicators.py`           | 📐 Indicateurs techniques réutilisables (EMA, SMA, ATR, RSI)         |
| `allocator.py`            | Répartit le capital Binance entre Trail Range, CrashBot et Listing Bot |
| `risk_manager.py`         | Calcul de taille de position, zero-risk, trailing, equity               |
| `position_store.py`       | Sérialisation/désérialisation des positions en JSON                  |
| `bot_binance.py`          | 🔄 Boucle principale Trail Range (Binance, ~284 paires, OCO)            |
| `bot_binance_crashbot.py` | 💥 Boucle principale CrashBot (Binance, dip-buy, step-trail)            |
| `bot_binance_listing.py`  | 🆕 Boucle principale Listing Bot (Binance, nouveaux listings, OCO)       |
| `listing_detector.py`     | 🆕 Détection listings, filtre momentum, niveaux OCO, force close      |
| `bot_london.py`           | 🇬🇧 Boucle principale London Breakout (Revolut X, 8 paires, maker-only)  |
| `bot_infinity.py`         | ♾️ Boucle principale Infinity (Revolut X, BTC+AAVE+XLM, DCA inversé) |
| `infinity_engine.py`      | ♾️ Logique DCA inversé : check_first_entry, paliers, trailing high   |
| `dca_engine.py`           | 📈 Logique DCA RSI v2 : brackets RSI, MVRV progressif, régime MA200, spending caps, DCADecision |
| `bot_dca.py`              | 📈 Boucle principale DCA RSI (Revolut X, BTC+ETH, maker-only)           |
| `onchain.py`              | 📈 Métriques on-chain (MVRV via CoinMetrics Community API, cache 1h) |
| `bot.py`                  | (legacy) Ancien bot Dow Theory Revolut X                                |
| `binance_client.py`       | Communique avec l'API Binance (OCO, market, balances)                   |
| `revolut_client.py`       | Communique avec l'API Revolut X (Ed25519, limit orders)                 |
| `data_provider.py`        | Récupère les bougies OHLCV                                            |
| `telegram.py`             | Envoie les alertes sur ton téléphone via Telegram                     |
| `trade_logger.py`         | Log chaque trade, heartbeat et allocation dans Firebase                 |
| `client.py` (firebase)    | Connexion Firestore, CRUD générique                                   |

### Dashboard

| Dashboard            | Port | Description                                           |
| -------------------- | ---- | ----------------------------------------------------- |
| 📊 Dashboard unifié | 8502 | 7 onglets : Overview, Trail Range, CrashBot, Listing, Infinity, London, DCA |

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

## Exemple concret — Trade LISTING

Imaginons qu'un nouveau token NEWUSDC est listé sur Binance :

### 1️⃣ Détection du listing

```
Symboles connus : 284 paires USDC
Poll exchangeInfo → 285 paires !
Nouveau symbole : NEWUSDC (status = TRADING)
→ 🔔 Nouveau listing détecté : NEWUSDC
```

### 2️⃣ Filtre momentum (klines 1m)

```
Kline 1m de NEWUSDC :
  OPEN = 1.2000$ | HIGH = 1.7040$
  Momentum = (1.7040 - 1.2000) / 1.2000 = +42% ≥ 30% ✅
→ 🆕 Momentum validé ! Achat autorisé.
```

### 3️⃣ Entrée market + OCO1

```
Capital Binance = 3 226 USDC | Listing Bot = 30% = 968 USDC
Slots libres : 3 → Capital/slot = 968 / 3 = 322 USDC

→ MARKET BUY NEW @ 1.2345$
  Taille = 322 / 1.2345 = 261 NEW
  OCO1 : SL = 1.2345 × 0.92 = 1.1357$ (-8%)
         TP1 = 1.2345 × 1.30 = 1.6049$ (+30%)
```

### 4️⃣ Prix monte → re-arm OCO2

```
Prix monte à 1.5728$ (≥ TP1 × 0.98 = 1.5728$) → Re-arm !
  Annule OCO1
  SL2 = 1.6049 × 0.769 = 1.2342$ (≈ breakeven !)
  TP2 = 1.6049 × 1.538 = 2.4683$ (+100%)
  Place OCO2 : SL2 + TP2
```

### 5️⃣ TP2 touché → gros gain

```
Prix explose à 2.4683$ → TP2 ✅
  → Vend 261 NEW @ 2.4683$
  → Gain : (2.4683 - 1.2345) × 261 = 322 USDC → +100% 🎉
```

### 6️⃣ Scénario alternatif : force close

```
Prix stagne à 1.10$ pendant 7 jours
→ ⏰ Force close ! Vente market @ 1.10$
→ Perte : (1.10 - 1.2345) × 261 = -35 USDC (-10.9%)
```

---

## Exemple concret — Trade LONDON BREAKOUT

Imaginons ETH-USD sur Revolut X :

### 1️⃣ Accumulation du range de session (08-16 UTC)

```
Bougie H4 08:00 : High = 3 520$ | Low = 3 480$
Bougie H4 12:00 : High = 3 545$ | Low = 3 490$

→ Session range : High = 3 545$ | Low = 3 480$
→ Largeur = (3545 - 3480) / 3480 = 1.87% ≥ 1.5% ✅
```

### 2️⃣ Breakout détecté (bougie H4 16:00)

```
Bougie H4 16:00 :
  Close: 3 560$ > session_high 3 545$ ✅
  Volume: 14.2M (moy 20: 5.8M → 2.45×) ✅
  Range session: 1.87% ≥ 1.5% ✅
→ 🇬🇧 BREAKOUT HAUSSIER détecté !
```

### 3️⃣ Entrée et gestion

```
→ 🇬🇧 SIGNAL LONDON BREAKOUT BUY !
  Ordre limit maker @ 3 560$ (0% frais)
  ATR(14) = 55$
  SL = 3 560 - 2.0 × 55 = 3 450$ (-3.1%)
  TP1 = 3 560 × 1.02 = 3 631$ (+2%) → vend 50%
  TP2 = 3 560 × 1.05 = 3 738$ (+5%) → vend le reste
  Risque 5% de 263$ = 13.15$
  Taille = 13.15 / (3560 - 3450) = 0.1195 ETH → cappé à 50%
```

### 4️⃣ TP1 touché → breakeven

```
Prix monte à 3 631$ → TP1 ✅
  → Vend 50% (0.0598 ETH) @ 3 631$ (+2%)
  → SL remonté à 3 560$ (breakeven)
```

### 5️⃣ TP2 touché → clôture complète

```
Prix monte à 3 738$ → TP2 ✅
  → Vend le reste (0.0597 ETH) @ 3 738$ (+5%)
  → ✅ Trade terminé ! PnL ≈ +4.63$ 🎉
  → Cooldown 8h sur ETH-USD
```

---

## Exemple concret — Trade INFINITY

Imaginons BTC-USD sur Revolut X :

### 1️⃣ Le bot surveille le trailing high

```
Trailing high (72 bougies H4 = 12 jours) = 74 064$
Prix actuel = 71 200$
Drop = -3.87% → < 5% → ❌ Pas encore
Cible d'entrée = 74 064 × 0.95 = 70 361$
→ ⏳ On attend. Écart : 839$ (1.2%)
```

### 2️⃣ Nouvelle bougie H4 — évaluation d'entrée

```
🕐 Bougie H4 de 08:00 UTC
Prix = 70 150$ (close H4)

  Drop = (74064 - 70150) / 74064 = -5.28% ≥ 5% ✅
  RSI(14) = 42 ≤ 50 ✅
→ ♾️ SIGNAL INFINITY BUY !
```

### 3️⃣ Premier achat (L1)

```
Capital Revolut X = 1 315$ | Allocation Infinity 65% = 855$

L1 : 25% du capital → 214$
→ Ordre limit maker @ 70 150$ (0% frais)
  Taille = 214 / 70 150 = 0.00305 BTC
  PMP = 70 150$ (1 seul achat)
  SL = 70 150 × 0.85 = 59 628$
```

### 4️⃣ Le prix continue de baisser → L2

```
Prix descend à 66 658$ → palier L2 atteint (-10%)
  L2 : 20% du capital → 171$
  → Achat 0.00257 BTC @ 66 658$
  PMP recalculé = (214 + 171) / (0.00305 + 0.00257) = 68 541$
  SL = 68 541 × 0.85 = 58 260$
```

### 5️⃣ Le prix remonte → Ventes par paliers

```
PMP = 68 541$ | BTC total = 0.00562

Prix remonte à 69 089$ → TP1 = PMP × 1.008 = 69 089$ ✅
  → Vend 20% (0.00112 BTC) @ 69 089$
  → Active breakeven stop @ PMP (68 541$)

Prix monte à 69 569$ → TP2 = PMP × 1.015 = 69 569$ ✅
  → Vend 20% (0.00112 BTC) @ 69 569$

Prix monte à 70 049$ → TP3 = PMP × 1.022 = 70 049$ ✅
  → Vend 20% (0.00112 BTC) @ 70 049$

Prix monte à 70 597$ → TP4 = PMP × 1.030 = 70 597$ ✅
  → Vend 20% (0.00112 BTC) @ 70 597$

Prix monte à 71 282$ → TP5 = PMP × 1.040 = 71 282$ ✅
  → Vend tout le reste (0.00114 BTC) @ 71 282$

→ ✅ Cycle terminé ! PnL ≈ +7.50$ 🎉
→ Retour en phase WAITING
```

### 6️⃣ Scénario alternatif : Stop Loss

```
PMP = 68 541$ | SL = 68 541 × 0.85 = 58 260$

Prix crash à 58 100$ → SL touché ❌
  → Vente market totale (taker 0.09%)
  → Perte ≈ -15% du capital investi
  → Retour en WAITING, compteur stops_consec +1
```

---

## Exemple concret — Trade DCA RSI

Imaginons le bot DCA v2 au jour 45 de son fonctionnement :

### 1️⃣ Achat quotidien normal (régime NORMAL)

```
🕐 10:00 UTC — Tick DCA v2
RSI(14) daily BTC = 48.2 | MVRV = 1.37 | Régime = NORMAL

  Bracket = NEUTRAL (45 ≤ 48.2 ≤ 55) → Base = 60$
  MVRV 1.37 ≥ 1.0 → ×1.0 → Final = 60$
  Regime NORMAL → BTC 90% / ETH 10%

📈 Achat BTC-USD :
  Montant : 60 × 0.90 = 54.00$
  Prix : 67 500$ | Size = 0.00080000 BTC
  → Ordre limit maker (0% frais) ✅

📈 Achat ETH-USD :
  Montant : 60 × 0.10 = 6.00$
  Prix : 3 400$ | Size = 0.00176471 ETH
  → Ordre limit maker (0% frais) ✅

Caps : mois $960/$1500 | sem $260/$400
Budget DCA restant : 3 876$ / 4 250$
```

### 2️⃣ RSI élevé → skip

```
🕐 10:00 UTC — Tick DCA v2
RSI(14) daily BTC = 73.5

  Bracket = OVERBOUGHT (73.5 > 70) → Montant = 0$
  → ⏭️ Pas d'achat aujourd'hui (marché trop cher)
```

### 3️⃣ MVRV boost en régime WEAK

```
🕐 10:00 UTC — Tick DCA v2
RSI(14) daily BTC = 42.0 | MVRV = 0.92 | Régime = WEAK (prix < MA200)

  Bracket = OVERSOLD (42.0 < 45) → Base = 90$
  MVRV 0.92 < 1.0 → ×1.5 → 90 × 1.5 = 135$
  Cap daily : min(135, 150) = 135$
  Check caps : mois $1200/$1500 → OK | sem $300/$400 → OK
  Boost cooldown : dernier boost > 24h → OK
  Regime WEAK → BTC 95% / ETH 5%

📈 Achat BTC-USD :
  Montant : 135 × 0.95 = 128.25$
  → Ordre limit maker ✅

📈 Achat ETH-USD :
  Montant : 135 × 0.05 = 6.75$
  → Ordre limit maker ✅
```

### 4️⃣ Crash reserve — BTC crashe

```
BTC chute brutalement. Crash anchor = max(high90j, high180j) = 72 000$

Prix actuel = 60 120$ → Drop = -16.5%

  Drop ≥ 15% → 🚨 CRASH LEVEL 1
  → Achat bonus 25% reserve = 492$ (BTC only)
  Size = 492 / 60 120 = 0.00818 BTC
  → Ordre limit maker ✅
  Crash reserve restante : 1 475$ / 1 967$

Prix continue à 53 280$ → Drop = -26.0%
  Drop ≥ 25% → 🚨 CRASH LEVEL 2
  → Achat bonus 35% reserve = 688$ (BTC only)
  → Crash reserve restante : 787$ / 1 967$
```

### 5️⃣ Récupération → reset crash levels

```
BTC remonte à 65 520$ → Drop = -9.0% (< 10%)
  → ✅ Crash levels réinitialisés
  → Les niveaux -15%, -25%, -35% pourront se redéclencher
```

### 6️⃣ Spending cap atteint

```
🕐 10:00 UTC — Tick DCA v2
RSI = 40.0 | MVRV = 0.88 → Base 90$ × 1.5 = 135$

  Check weekly cap : sem $380/$400 → Réduit à 20$
  → Achat pour 20$ seulement (cap protection)
  → DCADecision: cap_applied=true
```

---

## Ce que les bots ne font PAS

| ❌ Ne fait pas        | ✅ Fait à la place                                          |
| --------------------- | ------------------------------------------------------------ |
| Prédire l'avenir     | Constater des patterns (range, crash, impulsion) et réagir  |
| Miser tout le capital | Risk 2–5% par trade, allocation dynamique                   |
| Shorter en spot       | CrashBot et London = long only. Trail Range = long & short |
| Trader en permanence  | Chaque bot attend SES conditions spécifiques                |
| Ignorer les pertes    | Kill-switch mensuel, SL sur chaque trade, trailing lock      |

---

## Les paramètres importants (fichier `.env`)

### Trail Range 🔄

| Paramètre                          | Valeur   | Ce que ça fait                               |
| ----------------------------------- | -------- | --------------------------------------------- |
| `RISK_PERCENT`                    | 5%       | Risque par trade Range                        |
| `RANGE_WIDTH_MIN`                 | 2%       | Largeur minimum du range pour trader          |
| `RANGE_COOLDOWN_BARS`             | 3        | Bougies H4 de pause après un breakout (=12h) |
| `BINANCE_RANGE_TRAIL_SWAP_PCT`    | variable | Distance au TP pour déclencher le swap OCO   |
| `BINANCE_RANGE_TRAIL_STEP_PCT`    | ~1%      | Extension du TP par step                      |
| `BINANCE_RANGE_TRAIL_SL_LOCK_PCT` | ~2%      | Protection du profit verrouillé              |

### CrashBot 💥

| Paramètre                    | Valeur | Ce que ça fait                           |
| ----------------------------- | ------ | ----------------------------------------- |
| `drop_threshold`            | 20%    | Baisse minimum pour déclencher un signal |
| `lookback_bars`             | 12     | Fenêtre = 12 × 4h = 48 heures           |
| `tp_pct`                    | 8%     | Take profit initial                       |
| `atr_sl_mult`               | 1.5    | Multiplicateur ATR pour le SL             |
| `trail_step_pct`            | 0.5%   | Extension du TP par step                  |
| `BINANCE_CRASHBOT_KILL_PCT` | -10%   | Seuil du kill-switch mensuel              |
| `MOMENTUM_SIZING`           | true   | Active le sizing dynamique (W/L)          |
| `RISK_BOOST_MULT`           | ×1.2   | Multiplicateur risk après un WIN          |
| `RISK_SHRINK_MULT`          | ×0.8   | Multiplicateur risk après un LOSS         |
| `MIN_RISK_PCT`              | 2%     | Plancher du risk% dynamique               |
| `MAX_RISK_PCT`              | 10%    | Plafond du risk% dynamique                |

### Listing Bot 🆕

| Paramètre                          | Valeur  | Ce que ça fait                                |
| ----------------------------------- | ------- | ---------------------------------------------- |
| `LISTING_CAPITAL_PCT`             | 30%     | Part fixe du capital Binance pour le Listing   |
| `LISTING_MAX_SLOTS`               | 3       | Max 3 positions simultanées                    |
| `LISTING_MAX_ALLOC_USD`           | 5 000$  | Plafond par slot                               |
| `LISTING_SL_INIT_PCT`             | 8%      | Stop-loss initial sous l'entrée                |
| `LISTING_TP_INIT_PCT`             | 30%     | Take-profit initial au-dessus de l'entrée      |
| `LISTING_TP_NEAR_RATIO`           | 0.98    | Seuil re-arm OCO (98% du TP1)                 |
| `LISTING_SL2_TP1_MULT`            | 0.769   | SL2 = TP1 × 0.769 (≈ breakeven)              |
| `LISTING_TP2_TP1_MULT`            | 1.538   | TP2 = TP1 × 1.538 (+100%)                     |
| `LISTING_MOMENTUM_PCT`            | 30%     | Momentum minimum pour valider un listing       |
| `LISTING_MOMENTUM_WINDOW_MIN`     | 1 min   | Fenêtre kline pour mesurer le momentum         |
| `LISTING_HORIZON_DAYS`            | 7 jours | Force close si position ouverte > horizon      |
| `LISTING_POLL_INTERVAL_SECONDS`   | 10s     | Fréquence de polling principal                 |
| `LISTING_EXCHANGEINFO_CACHE_SECONDS` | 30s  | Cache exchangeInfo pour limiter les appels      |
| `LISTING_HEARTBEAT_SECONDS`       | 600s    | Heartbeat Telegram (10 min)                    |

### London Breakout 🇬🇧

| Paramètre                   | Valeur  | Ce que ça fait                         |
| ---------------------------- | ------- | --------------------------------------- |
| `LON_RISK_PERCENT`         | 5%      | Risque par trade                        |
| `LON_MAX_POSITIONS`        | 1       | Max 1 position simultanée              |
| `LON_SESSION_START_HOUR`   | 8       | Début session Londres (UTC)              |
| `LON_SESSION_END_HOUR`     | 16      | Fin session Londres (UTC)               |
| `LON_SL_ATR_MULT`          | 2.0     | Multiplicateur ATR pour le SL           |
| `LON_TP1_PCT`              | 2%      | TP1 : vente 50%                        |
| `LON_TP2_PCT`              | 5%      | TP2 : vente du reste                   |
| `LON_VOL_MULT`             | 2.0×   | Volume minimum (vs MA20)                |
| `LON_MIN_RANGE_PCT`        | 1.5%    | Largeur minimum du range de session     |
| `LON_POLLING_SECONDS`      | 30s     | Fréquence de vérification             |
| `LON_MAKER_WAIT_SECONDS`   | 60s     | Temps d'attente pour un fill maker      |

### Infinity ♾️

| Paramètre                   | Valeur                        | Ce que ça fait                                 |
| ---------------------------- | ----------------------------- | ----------------------------------------------- |
| `INF_TRADING_PAIRS`        | BTC-USD,AAVE-USD,XLM-USD     | Paires tradées (configs validées par paire)   |
| `INF_CAPITAL_PCT`          | 80%                           | Part du capital Revolut X allouée (÷3 paires) |
| `INF_POLLING_SECONDS`      | 30s                           | Fréquence de polling prix                      |
| `INF_MAKER_WAIT_SECONDS`   | 60s                           | Attente max pour fill maker                     |
| `INF_HEARTBEAT_SECONDS`    | 600s                          | Fréquence heartbeat (10 min)                   |

> Les paramètres de stratégie (buy_levels, sell_levels, SL, trailing) sont codés en dur dans `PAIR_CONFIGS` (walk-forward validés).

| Paire    | Entry Drop | Trail High | SL   | Buy Levels            | Sell Levels                   |
| -------- | ---------- | ---------- | ---- | --------------------- | ----------------------------- |
| BTC-USD  | 5%         | 72 bars    | 15%  | -5%→-25%              | +0.8%→+4.0%                   |
| AAVE-USD | 12%        | 48 bars    | 25%  | -12%→-42%             | +2%→+12%                      |
| XLM-USD  | 12%        | 48 bars    | 25%  | -12%→-42%             | +0.8%→+4.0%                   |

### Allocation dynamique

| Paramètre     | Valeur | Ce que ça fait                              |
| -------------- | ------ | -------------------------------------------- |
| `PF_LOW`     | 0.9    | Seuil bas du PF (en-dessous → Défensif)    |
| `PF_HIGH`    | 1.1    | Seuil haut du PF (au-dessus → Agressif)     |
| `MIN_TRADES` | 20     | Nombre minimum de trades pour évaluer le PF |

### DCA RSI v2 📈

| Paramètre                        | Valeur     | Ce que ça fait                                |
| ---------------------------------- | ---------- | ---------------------------------------------- |
| `DCA_CAPITAL_PCT`                | 1.0 (100%) | Part du solde Revolut X allouée au DCA         |
| `DCA_ACTIVE_PCT`                 | 0.85 (85%) | Part du capital DCA → achats quotidiens        |
| `DCA_CRASH_PCT`                  | 0.15 (15%) | Part du capital DCA → crash reserve            |
| `DCA_BASE_DAILY_AMOUNT`          | 30$        | Montant de base (bracket WARM, RSI 55-70)      |
| `DCA_MAX_DAILY_BUY`             | 150$       | Plafond journalier absolu                      |
| `DCA_BTC_ALLOC`                  | 90%        | Allocation BTC (régime NORMAL)                 |
| `DCA_ETH_ALLOC`                  | 10%        | Allocation ETH (régime NORMAL)                 |
| `DCA_RSI_OVERBOUGHT`            | 70         | Seuil RSI overbought (skip)                    |
| `DCA_RSI_WARM`                   | 55         | Seuil RSI warm (montant ×1)                    |
| `DCA_RSI_NEUTRAL_LOW`           | 45         | Seuil RSI oversold (montant ×3)                |
| `DCA_MONTHLY_CAP`               | 1500$      | Spending cap mensuel                           |
| `DCA_WEEKLY_CAP`                 | 400$       | Spending cap hebdomadaire                      |
| `DCA_BOOST_COOLDOWN_HOURS`      | 24h        | Cooldown entre deux boosts MVRV                |
| `DCA_BOOST_THRESHOLD`           | 120$       | Seuil pour activer le cooldown                 |
| `DCA_REGIME_FILTER_ENABLED`     | true       | Active le filtre régime MA200                  |
| `DCA_CAPITULATION_THRESHOLD`    | 0.85       | Ratio prix/MA200 pour CAPITULATION             |
| `DCA_MVRV_ENABLED`              | true       | Active le signal MVRV on-chain                 |
| `DCA_MVRV_THRESHOLD`            | 1.0        | Seuil MVRV pour boost ×1.5                    |
| `DCA_MVRV_DEEP_THRESHOLD`      | 0.85       | Seuil MVRV pour boost ×2.0                    |
| `DCA_MVRV_MULT_LOW`            | 1.5        | Multiplicateur si MVRV < 1.0                   |
| `DCA_MVRV_MULT_DEEP`           | 2.0        | Multiplicateur si MVRV < 0.85                  |
| `DCA_CRASH_DROP_1/2/3`          | 15/25/35%  | Seuils de drop pour crash reserve              |
| `DCA_CRASH_ANCHOR_LONG_DAYS`   | 180 jours  | Fenêtre longue pour crash anchor               |
| `DCA_CRASH_BTC_ONLY`            | true       | Crash reserve 100% BTC                         |
| `DCA_CRASH_LOOKBACK_DAYS`       | 90 jours   | Fenêtre courte pour le rolling high            |
| `DCA_EXECUTION_HOUR_UTC`        | 10         | Heure d'exécution quotidienne (UTC)            |
| `DCA_POLLING_SECONDS`           | 60s        | Fréquence de polling                           |
| `DCA_HEARTBEAT_SECONDS`         | 600s       | Heartbeat Telegram (10 min)                    |
| `DCA_MAKER_WAIT_SECONDS`        | 60s        | Attente max pour fill maker                    |

---

## Infrastructure & Déploiement

### VPS (Contabo)

|                     | Détail              |
| ------------------- | -------------------- |
| **OS**        | Ubuntu 22.04 LTS     |
| **IP**        | 213.199.41.168       |
| **Connexion** | `ssh BOT-VPS`      |
| **App**       | `/opt/tradex`      |
| **Python**    | 3.10, venv `.venv` |
| **Gestion**   | systemd services     |

### Services actifs

| Service                      | Description                                     | Port |
| ---------------------------- | ----------------------------------------------- | ---- |
| `tradex-binance`           | Bot Trail Range (284 paires USDC, OCO)          | —   |
| `tradex-binance-crashbot`  | Bot CrashBot (284 paires USDC, dip-buy)         | —   |
| `tradex-listing`           | Bot Listing (nouveaux listings USDC, OCO dynamique) | —   |
| `tradex-london`          | Bot London Breakout (8 paires USD, Revolut X)   | —   |
| `tradex-infinity`          | Bot Infinity (BTC+AAVE+XLM, DCA inversé, Revolut X)   | —   |
| `tradex-dca`               | Bot DCA RSI v2 (BTC+ETH, RSI+MVRV+regime+caps, Revolut X) | —   |
| `tradex-dashboard-unified` | Dashboard Streamlit unifié                     | 8502 |
| `tradex-dca-dashboard`     | Dashboard DCA standalone (Plotly)               | 8503 |
| `tradex-telegram-commands` | Bot Telegram de pilotage opérateur             | —   |

### Commandes utiles

```bash
# Logs en direct
ssh BOT-VPS 'sudo journalctl -u tradex-binance -f'
ssh BOT-VPS 'sudo journalctl -u tradex-binance-crashbot -f'
ssh BOT-VPS 'sudo journalctl -u tradex-listing -f'
ssh BOT-VPS 'sudo journalctl -u tradex-london -f'
ssh BOT-VPS 'sudo journalctl -u tradex-infinity -f'
ssh BOT-VPS 'sudo journalctl -u tradex-dca -f'

# État de tous les services
ssh BOT-VPS 'for svc in tradex-binance tradex-binance-crashbot tradex-listing tradex-london tradex-infinity tradex-dca tradex-dashboard-unified tradex-dca-dashboard; do echo -n "$svc: "; sudo systemctl is-active $svc; done'

# Redémarrer un bot
ssh BOT-VPS 'sudo systemctl restart tradex-binance'

# Dashboard
# http://213.199.41.168:8502
```

### Commandes Telegram opérateur

Le service `tradex-telegram-commands` permet le pilotage à chaud via Firestore (`runtime_overrides` et `runtime_actions`) sans redémarrage des bots.

- Heartbeat live: `/hb get`, `/hb set <bot|all> <sec>`, `/hb reset <bot|all>`
- Observabilité: `/health`, `/alloc`, `/open <bot|all>`, `/last <bot|all> [n]`, `/cmdlog [n]`, `/logs <bot|all> [n]`
- Actions manuelles: `/close now <bot> <symbol|all>`, `/sl set <bot> <symbol> <price>`, `/tp set <bot> <symbol> <price>`

#### Sécurité — double confirmation

La commande de fermeture globale `/close now <bot> all` exige une validation explicite en 2 étapes:

1. `/close now <bot> all` → retourne un token temporaire
2. `/confirm <token>` dans les 120 secondes → action réellement enfilée dans `runtime_actions`

Si le token expire, aucune fermeture globale n'est enfilée et la demande doit être relancée.
