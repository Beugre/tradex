# 🚀 Déploiement TradeX sur VPS

## Prérequis

- VPS **Ubuntu 22.04+** ou **Debian 12+**
- Accès SSH configuré (alias `vps-connexion` ou clé SSH)
- Python 3.9+ disponible sur le VPS

## Architecture du déploiement

```
/opt/tradex/              ← Répertoire de l'app sur le VPS
├── src/                  ← Code source
├── .venv/                ← Environnement virtuel Python
├── .env                  ← Configuration (clés API, paramètres)  ⚠️ NON versionné
├── private.pem           ← Clé Ed25519 Revolut X                 ⚠️ NON versionné
├── requirements.txt
└── logs/
```

Le bot tourne en tant que service **systemd** (`tradex.service`), avec redémarrage automatique en cas de crash.

---

## 🔧 Première installation

### 1. Configurer l'accès SSH

Si ce n'est pas fait, ajouter dans `~/.ssh/config` :

```
Host vps
    HostName <IP_DU_VPS>
    User root
    IdentityFile ~/.ssh/id_ed25519
```

### 2. Lancer le setup sur le VPS

```bash
# Depuis ta machine locale, dans le dossier du projet :
scp -r . vps:/tmp/tradex-install
ssh vps 'chmod +x /tmp/tradex-install/deploy/setup-vps.sh && sudo /tmp/tradex-install/deploy/setup-vps.sh'
```

Ce script :
- Installe Python 3 et les paquets système
- Crée un utilisateur dédié `tradex`
- Crée `/opt/tradex` avec un environnement virtuel
- Installe les dépendances
- Configure le service systemd

### 3. Transférer les fichiers sensibles

```bash
# Copier la clé privée Ed25519
scp private.pem vps:/opt/tradex/private.pem
ssh vps 'chmod 600 /opt/tradex/private.pem && chown tradex:tradex /opt/tradex/private.pem'

# Copier le .env (ou l'éditer directement sur le VPS)
scp .env vps:/opt/tradex/.env
ssh vps 'chmod 600 /opt/tradex/.env && chown tradex:tradex /opt/tradex/.env'
```

### 4. Démarrer le bot

```bash
ssh vps 'sudo systemctl start tradex'
```

---

## 🔄 Mises à jour (déploiement courant)

Après chaque modification du code :

```bash
./deploy/deploy.sh
```

Ce script :
1. Synchronise le code via `rsync` (exclut `.env`, `private.pem`, `.venv`, tests)
2. Met à jour les dépendances pip
3. Redémarre le service
4. Vérifie que le bot est bien actif

⚠️ Important : les scripts de déploiement utilisent `rsync --delete` avec des règles de protection pour préserver les répertoires runtime du VPS (`/opt/tradex/.venv`, `/opt/tradex/data`, `/opt/tradex/logs`).

Évite les commandes `rsync --delete` manuelles hors scripts `deploy/*.sh` si ces protections ne sont pas présentes.

> 💡 Pour changer l'hôte SSH, modifier `VPS_HOST` dans le script ou l'exporter :
> ```bash
> VPS_HOST=user@1.2.3.4 ./deploy/deploy.sh
> ```

---

## 📊 Surveillance

### Logs en temps réel
```bash
ssh vps 'sudo journalctl -u tradex -f'
```

### Dernières 50 lignes
```bash
ssh vps 'sudo journalctl -u tradex -n 50 --no-pager'
```

### Statut du service
```bash
ssh vps 'sudo systemctl status tradex'
```

### Redémarrer
```bash
ssh vps 'sudo systemctl restart tradex'
```

### Arrêter
```bash
ssh vps 'sudo systemctl stop tradex'
```

---

## 🔐 Sécurité

- Le bot tourne sous l'utilisateur système `tradex` (pas root)
- `.env` et `private.pem` ont les permissions `600` (lecture owner uniquement)
- Le service systemd est durci : `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`
- Les fichiers sensibles ne sont **jamais** synchronisés par `deploy.sh`

---

## 🆘 Dépannage

| Problème | Solution |
|----------|----------|
| Le service ne démarre pas | `sudo journalctl -u tradex -n 30` pour voir l'erreur |
| `ModuleNotFoundError` | `sudo -u tradex /opt/tradex/.venv/bin/pip install -r /opt/tradex/requirements.txt` |
| Erreur de signature API | Vérifier que `private.pem` est bien copié et correspond à la clé publique sur Revolut X |
| `.env` non trouvé | Copier depuis `.env.example` : `cp .env.example .env && nano .env` |
| Bot redémarre en boucle | Vérifier les logs, probable erreur de config ou clé API invalide |
