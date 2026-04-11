"""WindCast configuration — Pydantic Settings with env override support."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    """Per-dataset metadata."""

    dataset_id: str
    rated_power_kw: float
    hub_height_m: float
    rotor_diameter_m: float
    n_turbines: int
    latitude: float
    longitude: float
    train_years: int = 5
    val_years: int = 1


KELMARSH = DatasetConfig(
    dataset_id="kelmarsh",
    rated_power_kw=2050.0,
    hub_height_m=78.5,
    rotor_diameter_m=92.0,
    n_turbines=6,
    latitude=52.4016,
    longitude=-0.9436,
)

HILL_OF_TOWIE = DatasetConfig(
    dataset_id="hill_of_towie",
    rated_power_kw=2300.0,
    hub_height_m=80.0,
    rotor_diameter_m=82.0,
    n_turbines=21,
    latitude=57.34,
    longitude=-2.65,
)

PENMANSHIEL = DatasetConfig(
    dataset_id="penmanshiel",
    rated_power_kw=2050.0,
    hub_height_m=78.5,
    rotor_diameter_m=82.0,
    n_turbines=13,
    latitude=55.905,
    longitude=-2.29,
)


class DemandDatasetConfig(BaseModel):
    """Per-dataset metadata for demand forecasting."""

    dataset_id: str
    zone_id: str
    population: int | None = None
    latitude: float
    longitude: float
    timezone: str
    train_years: int = 5
    val_years: int = 1


SPAIN_DEMAND = DemandDatasetConfig(
    dataset_id="spain_demand",
    zone_id="ES",
    population=47_000_000,
    latitude=40.4168,
    longitude=-3.7038,
    timezone="Europe/Madrid",
    train_years=2,
    val_years=1,
)

RTE_FRANCE = DemandDatasetConfig(
    dataset_id="rte_france",
    zone_id="FR",
    population=68_000_000,
    latitude=48.8566,  # Paris
    longitude=2.3522,
    timezone="Europe/Paris",
    train_years=8,
    val_years=2,
)


class SolarDatasetConfig(BaseModel):
    """Per-dataset metadata for solar PV forecasting."""

    dataset_id: str
    system_id: str
    capacity_kw: float
    tilt_deg: float
    azimuth_deg: float
    latitude: float
    longitude: float
    timezone: str
    train_years: int = 5
    val_years: int = 1


PVDAQ_SYSTEM4 = SolarDatasetConfig(
    dataset_id="pvdaq_system4",
    system_id="4",
    capacity_kw=2.2,
    tilt_deg=40.0,
    azimuth_deg=180.0,
    latitude=39.7407,
    longitude=-105.1686,
    timezone="America/Denver",
)

DATASETS: dict[str, DatasetConfig | DemandDatasetConfig | SolarDatasetConfig] = {
    "kelmarsh": KELMARSH,
    "hill_of_towie": HILL_OF_TOWIE,
    "penmanshiel": PENMANSHIEL,
    "spain_demand": SPAIN_DEMAND,
    "rte_france": RTE_FRANCE,
    "pvdaq_system4": PVDAQ_SYSTEM4,
}


class QCConfig(BaseModel):
    """Quality control thresholds."""

    max_wind_speed_ms: float = 40.0
    max_gap_fill_minutes: int = 30
    frozen_sensor_threshold_minutes: int = 60
    rated_power_tolerance: float = 1.05
    min_pitch_curtailment_deg: float = 3.0


class DemandQCConfig(BaseModel):
    """QC thresholds for demand data."""

    max_load_mw: float = 100_000.0  # France peak is ~90 GW; 100 GW headroom covers Spain too
    min_load_mw: float = 10_000.0
    max_temperature_c: float = 50.0
    min_temperature_c: float = -20.0
    max_wind_speed_ms: float = 50.0
    max_gap_fill_hours: int = 3


class SolarQCConfig(BaseModel):
    """QC thresholds for solar data."""

    max_power_kw: float = 5.0
    max_irradiance_wm2: float = 1500.0
    min_irradiance_wm2: float = -10.0
    max_temperature_c: float = 60.0
    min_temperature_c: float = -30.0
    max_gap_fill_intervals: int = 4
    nighttime_power_threshold_kw: float = 0.01


DOMAIN_RESOLUTION: dict[str, int] = {
    "wind": 10,  # 10-min SCADA
    "demand": 60,  # Hourly
    "solar": 15,  # 15-min PVDAQ
}
"""Data resolution in minutes per domain. Used to convert horizon steps → timedelta."""


class WindCastSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WINDCAST_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    dataset_id: str = "kelmarsh"
    domain: str = "wind"
    data_resolution_minutes: int = 10
    train_years: int = 5
    val_years: int = 1
    test_years: int = 1
    forecast_horizons: list[int] = [1, 6, 12, 24, 48]
    qc: QCConfig = Field(default_factory=QCConfig)
    demand_qc: DemandQCConfig = Field(default_factory=DemandQCConfig)
    solar_qc: SolarQCConfig = Field(default_factory=SolarQCConfig)
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # RTE API credentials (optional — live actuals on dashboard)
    rte_client_id: str | None = None
    rte_client_secret: str | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def dataset_config(self) -> DatasetConfig | DemandDatasetConfig | SolarDatasetConfig:
        return DATASETS[self.dataset_id]


@lru_cache(maxsize=1)
def get_settings() -> WindCastSettings:
    return WindCastSettings()
