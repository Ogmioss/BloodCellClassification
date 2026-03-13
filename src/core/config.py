"""Configuration centralisee des URLs de services et utilitaires partages."""

import os


# URLs internes (pour les health checks depuis les containers)
_INTERNAL_ENV_KEYS = {
    "api": "FASTAPI_URL",
    "mlflow": "MLFLOW_URL",
    "airflow": "AIRFLOW_URL",
    "grafana": "GRAFANA_URL",
    "prometheus": "PROMETHEUS_URL",
    "minio": "MINIO_URL",
}

# URLs externes (pour les liens navigateur) — suffixe _EXTERNAL_URL
_EXTERNAL_ENV_KEYS = {
    "api": "FASTAPI_EXTERNAL_URL",
    "mlflow": "MLFLOW_EXTERNAL_URL",
    "airflow": "AIRFLOW_EXTERNAL_URL",
    "grafana": "GRAFANA_EXTERNAL_URL",
    "prometheus": "PROMETHEUS_EXTERNAL_URL",
    "minio": "MINIO_EXTERNAL_URL",
}

_DEFAULTS = {
    "api": "http://localhost:8000",
    "mlflow": "http://localhost:5000",
    "airflow": "http://localhost:8080",
    "grafana": "http://localhost:3000",
    "prometheus": "http://localhost:9090",
    "minio": "http://localhost:9001",
}


def get_service_url(service: str) -> str:
    """Retourne l'URL interne d'un service (pour les appels inter-containers).

    Args:
        service: Nom du service (api, mlflow, airflow, grafana, prometheus).

    Returns:
        URL du service avec valeur par defaut pour le dev local.
    """
    env_key = _INTERNAL_ENV_KEYS.get(service)
    if env_key is None:
        msg = f"Service inconnu: {service}. Valides: {list(_DEFAULTS)}"
        raise ValueError(msg)
    return os.getenv(env_key, _DEFAULTS[service])


def get_service_external_url(service: str) -> str:
    """Retourne l'URL externe d'un service (pour le navigateur).

    Utilise *_EXTERNAL_URL si defini, sinon tombe sur l'URL interne,
    sinon le default localhost.
    """
    ext_key = _EXTERNAL_ENV_KEYS.get(service)
    if ext_key is None:
        msg = f"Service inconnu: {service}. Valides: {list(_DEFAULTS)}"
        raise ValueError(msg)
    ext_url = os.getenv(ext_key)
    if ext_url:
        return ext_url
    # Fallback sur l'URL interne puis le default
    return get_service_url(service)


def get_all_service_urls() -> dict[str, str]:
    """Retourne un dict avec toutes les URLs internes de services."""
    return {svc: get_service_url(svc) for svc in _DEFAULTS}


def get_all_external_urls() -> dict[str, str]:
    """Retourne un dict avec toutes les URLs externes (navigateur)."""
    return {svc: get_service_external_url(svc) for svc in _DEFAULTS}
