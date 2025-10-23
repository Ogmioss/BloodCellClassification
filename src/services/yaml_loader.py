from __future__ import annotations

from pathlib import Path
import yaml


class YamlLoader:
    """
    Classe pour charger et valider les variables de configuration depuis un fichier YAML.
    """
    
    def __init__(self, config_path: Path | str | None = None, project_root: Path | str | None = None):
        """
        Initialise le loader YAML.
        
        Args:
            config_path: Chemin vers le fichier de configuration YAML. 
                        Par défaut: conf.yaml à la racine du projet.
            project_root: Racine du projet. Par défaut: deux niveaux au-dessus du fichier yaml_loader.py.
        """
        if project_root is None:
            # yaml_loader.py is in src/services/, so project root is 2 levels up
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()
            
        if config_path is None:
            self.config_path = self.project_root / "conf.yaml"
        else:
            self.config_path = Path(config_path)
            if not self.config_path.is_absolute():
                self.config_path = self.project_root / self.config_path
                
        self._config_data: dict | None = None
        
    def _load_config(self) -> dict:
        """Charge le fichier de configuration YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Fichier de config introuvable: {self.config_path}")
        
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            
        if not isinstance(data, dict):
            raise ValueError("Le fichier YAML doit contenir un mapping clé/valeur à la racine.")
            
        return data
    
    @property
    def config(self) -> dict:
        """Retourne les données de configuration (chargées une seule fois)."""
        if self._config_data is None:
            self._config_data = self._load_config()
        return self._config_data
    
    def _resolve_dir(self, path_like: str | Path) -> Path:
        """Résout un chemin relativement à la racine du projet."""
        p = Path(path_like)
        return (self.project_root / p).resolve() if not p.is_absolute() else p.resolve()
    
    def _require_dir(self, path: Path, name: str) -> Path:
        """Valide qu'un chemin existe et est un dossier."""
        if not path.exists():
            raise FileNotFoundError(f"{name} n'existe pas: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"{name} n'est pas un dossier: {path}")
        return path
    
    def get_nested_value(self, path: str, default: any = None) -> any:
        """
        Récupère une valeur depuis une structure YAML imbriquée en utilisant la notation pointée.
        
        Args:
            path: Chemin vers la valeur (ex: "paths.data.root")
            default: Valeur par défaut si le chemin n'existe pas
            
        Returns:
            Valeur trouvée ou valeur par défaut
        """
        keys = path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current
    
    def get_dir(self, path: str, default: str | Path | None = None) -> Path:
        """
        Récupère et valide un chemin de dossier depuis la configuration hiérarchique.
        
        Args:
            path: Chemin vers la valeur dans la config (ex: "paths.data.root")
            default: Valeur par défaut si le chemin n'existe pas
            
        Returns:
            Path absolu validé du dossier
        """
        path_value = self.get_nested_value(path, default)
        
        if path_value is None:
            raise KeyError(f"Chemin '{path}' introuvable dans la configuration")
            
        resolved_path = self._resolve_dir(path_value)
        return self._require_dir(resolved_path, path)
    
    @property
    def data_dir(self) -> Path:
        """Retourne le chemin du dossier data."""
        return self.get_dir("paths.data.root", "./src/data")

    @property
    def data_processed_dir(self) -> Path:
        """Retourne le chemin du dossier data/processed."""
        return self.get_dir("paths.data.processed", "./src/data/processed")

    @property
    def data_raw_dir(self) -> Path:
        """Retourne le chemin du dossier data/raw."""
        return self.get_dir("paths.data.raw", "./src/data/raw")

    @property
    def notebooks_dir(self) -> Path:
        """Retourne le chemin du dossier notebooks."""
        return self.get_dir("paths.notebooks", "./notebooks")
    
    @property
    def services_dir(self) -> Path:
        """Retourne le chemin du dossier services."""
        return self.get_dir("paths.services", "./services")

    @property
    def logs_dir(self) -> Path:
        """Retourne le chemin du dossier logs."""
        return self.get_dir("paths.logs", "./logs")
    
    def get_all_paths(self) -> dict[str, Path]:
        """
        Retourne tous les chemins de dossiers configurés.
        """
        return {
            "DATA_DIR": self.data_dir,
            "DATA_RAW_DIR": self.data_raw_dir,
            "NOTEBOOKS_DIR": self.notebooks_dir,
            "SERVICES_DIR": self.services_dir,
        }
    
    def get_project_info(self) -> dict[str, str]:
        """
        Retourne les informations du projet depuis la configuration.
        """
        return {
            "name": self.get_nested_value("project.name", "completion-cablot"),
            "version": self.get_nested_value("project.version", "0.1.0"),
            "python_version": self.get_nested_value("environment.python_version", ">=3.11"),
        }


# Instance par défaut pour compatibilité avec l'ancienne API
_default_loader = YamlLoader()

# Variables au niveau module pour compatibilité
DATA_DIR: Path = _default_loader.data_dir
DATA_RAW_DIR: Path = _default_loader.data_raw_dir



def get_paths() -> dict[str, Path]:
    """
    Retourne les chemins résolus (absolus) sous forme de dict.
    Fonction de compatibilité avec l'ancienne API.
    """
    return _default_loader.get_all_paths()