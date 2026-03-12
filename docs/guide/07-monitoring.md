# 07 — Monitoring

Le monitoring repose sur le trio **Prometheus + Grafana + Pushgateway** pour couvrir à la fois les métriques de l'API et celles de l'entraînement.

## Architecture du monitoring

```
┌──────────────┐                          ┌────────────┐
│  FastAPI     │──── /metrics (pull) ────▶│ Prometheus │
│  (API)       │                          │            │
└──────────────┘                          │  scrape    │
                                          │  toutes    │
┌──────────────┐                          │  les 10s   │───▶ ┌─────────┐
│ TrainingService│── push (par epoch) ──▶│            │     │ Grafana │
│              │        ┌────────────┐    │            │     │         │
└──────────────┘        │Pushgateway │───▶│            │     └─────────┘
                        └────────────┘    └────────────┘
                                                │
┌──────────────┐                                │
│  MinIO       │──── /minio/v2/metrics ────────▶│
└──────────────┘        (pull, 30s)
```

**Deux modes de collecte** :
- **Pull** (standard) : Prometheus scrape les endpoints `/metrics` de l'API et de MinIO
- **Push** : Le TrainingService pousse les métriques d'epoch vers le Pushgateway (car l'entraînement est un batch, pas un service permanent)

## Métriques collectées

### Métriques API (automatiques via prometheus-fastapi-instrumentator)

| Métrique | Type | Description |
|----------|------|-------------|
| `http_requests_total` | Counter | Nombre de requêtes HTTP par route et status |
| `http_request_duration_seconds` | Histogram | Latence des requêtes |
| `http_requests_in_progress` | Gauge | Requêtes en cours |

### Métriques métier (custom, `src/api/metrics.py`)

| Métrique | Type | Description |
|----------|------|-------------|
| `bloodcell_predictions_total` | Counter | Prédictions par classe |
| `bloodcell_prediction_errors_total` | Counter | Erreurs de prédiction |
| `bloodcell_prediction_confidence` | Histogram | Distribution de confiance |
| `bloodcell_prediction_latency_ms` | Histogram | Latence de prédiction en ms |

### Métriques d'entraînement (via Pushgateway, `src/services/training_metrics.py`)

| Métrique | Type | Description |
|----------|------|-------------|
| `training_epoch_loss` | Gauge | Loss par epoch |
| `training_epoch_accuracy` | Gauge | Accuracy par epoch |
| `training_val_loss` | Gauge | Loss de validation |
| `training_val_accuracy` | Gauge | Accuracy de validation |

## Configuration Prometheus (`monitoring/prometheus.yml`)

```yaml
scrape_configs:
  - job_name: 'bloodcell-api'
    scrape_interval: 10s
    metrics_path: /metrics
    static_configs:
      - targets: ['api:8000']

  - job_name: 'pushgateway'
    scrape_interval: 10s
    static_configs:
      - targets: ['pushgateway:9091']

  - job_name: 'minio'
    scrape_interval: 30s
    metrics_path: /minio/v2/metrics/cluster
    static_configs:
      - targets: ['minio:9000']
```

## Dashboards Grafana

### Dashboard API (`monitoring/grafana/dashboards/bloodcell-api.json`)

Panneaux principaux :
- **Taux de requêtes** : requêtes/seconde par endpoint
- **Latence** : p50, p90, p99 des temps de réponse
- **Taux d'erreur** : pourcentage de réponses 4xx/5xx
- **Prédictions par classe** : distribution des classes prédites
- **Confiance** : histogramme de la confiance des prédictions
- **Informations modèle** : version, source, device

### Dashboard Training (`monitoring/grafana/dashboards/bloodcell-training.json`)

Panneaux principaux :
- **Loss par epoch** : train vs validation
- **Accuracy par epoch** : train vs validation
- **Durée d'entraînement** : temps total et par epoch
- **Performance par classe** : métriques détaillées

## Alertes (`monitoring/prometheus/alerts.yml`)

| Alerte | Condition | Sévérité | Signification |
|--------|-----------|----------|---------------|
| `APIDown` | API injoignable > 2 min | Critical | L'API ne répond plus |
| `HighPredictionErrorRate` | Taux d'erreur > 10% sur 5 min | Warning | Problème de qualité des prédictions |
| `LowConfidencePredictions` | Confiance < 0.5 pour > 30% des prédictions | Warning | Le modèle est peu sûr de lui |
| `TrainingStalled` | Pas de progrès > 30 min | Warning | L'entraînement est bloqué |

## Accès aux interfaces

| Service | URL | Identifiants |
|---------|-----|-------------|
| Prometheus | `http://localhost:9090` | — |
| Grafana | `http://localhost:3000` | admin / admin |
| Pushgateway | `http://localhost:9091` | — |

## Provisionnement automatique

Grafana est configuré pour se provisionner automatiquement au démarrage :
- **Datasources** : Prometheus est ajouté via `monitoring/grafana/provisioning/datasources/datasources.yml`
- **Dashboards** : les JSON sont montés depuis `monitoring/grafana/dashboards/`

Aucune configuration manuelle n'est nécessaire après `docker compose up`.

## Requêtes PromQL utiles

```promql
# Taux de prédictions par seconde
rate(bloodcell_predictions_total[5m])

# Latence p99 des prédictions
histogram_quantile(0.99, rate(bloodcell_prediction_latency_ms_bucket[5m]))

# Confiance moyenne
rate(bloodcell_prediction_confidence_sum[5m]) / rate(bloodcell_prediction_confidence_count[5m])

# Taux d'erreur
rate(bloodcell_prediction_errors_total[5m]) / rate(bloodcell_predictions_total[5m])
```
