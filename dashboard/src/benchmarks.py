"""Benchmark definitions shared by the dashboard and static reports.

One active benchmark at a time; static reports default to MSCI World.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    key: str
    ticker: str
    short_label: str
    label: str
    description: str
    color: str
    dash: str = "dot"


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "sp500_eur_hedged": BenchmarkConfig(
        key="sp500_eur_hedged",
        ticker="IUSE.L",
        short_label="S&P 500 cubierto EUR",
        label="IUSE.L — iShares S&P 500 con cobertura EUR UCITS ETF",
        description="Renta variable de Estados Unidos con cobertura en euros.",
        color="#64748B",
        dash="dot",
    ),
    "msci_world": BenchmarkConfig(
        key="msci_world",
        ticker="IWDA.AS",
        short_label="MSCI World",
        label="IWDA.AS — iShares Core MSCI World UCITS ETF",
        description="Renta variable global de países desarrollados.",
        color="#2563EB",
        dash="dash",
    ),
}

DEFAULT_REPORT_BENCHMARK_KEY = "msci_world"


def get_benchmark_config(key: str) -> BenchmarkConfig:
    """Return a benchmark config, falling back to the report default."""
    return BENCHMARKS.get(key, BENCHMARKS[DEFAULT_REPORT_BENCHMARK_KEY])


def benchmark_label_options(include_none: bool = True) -> dict[str, str]:
    """Mapping used by Streamlit selectors: display label → benchmark key."""
    options = {cfg.short_label: key for key, cfg in BENCHMARKS.items()}
    if include_none:
        return {"Sin índice": "none", **options}
    return options
