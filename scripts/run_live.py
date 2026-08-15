"""
Entry point en vivo: pipeline en el ultimo dia -> hoja de operativa.

Salida: outputs/live/ (Operativa_*.xlsx, posiciones_*.xlsx, execution_log.csv).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.data_loader import download_market_data
from src.pipeline import run_pipeline
from src.portfolio.audit_log import append_ejecucion_row
from src.portfolio.email_operaciones import send_operaciones_excel
from src.portfolio.positions_io import (
    ensure_positions_template,
    load_positions_from_excel,
    portfolio_value_eur,
    save_positions_to_excel,
    weights_from_positions,
)
from src.portfolio.registrador import run_registrador
import src.config as cfg


def _get_risk_tickers(market_data: dict) -> list[str]:
    metadata_by_ticker = {item["ticker"]: item for item in market_data["metadata"]}
    return [
        ticker for ticker in market_data["tickers"]
        if metadata_by_ticker.get(ticker, {}).get("role") != "defensive"
    ]


def run_single(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    current_positions: dict | None = None,
) -> dict:
    """Pipeline en el ultimo dia disponible -> Excel de operativa."""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        # Reproducibilidad: hoy por defecto es no determinista; se traza.
        print(f"[run_single] end_date no especificado — usando hoy: {end_date}")

    market_data = download_market_data(start_date=start_date, end_date=end_date)
    last_dt = market_data["prices"].index[-1]
    full_prices_row = market_data["prices"].loc[last_dt]

    risk_tickers = _get_risk_tickers(market_data)
    returns_df = market_data["returns"][risk_tickers]
    prices_df = market_data["prices"][risk_tickers]

    positions_path = Path(getattr(cfg, "POSITIONS_EXCEL_PATH", "outputs/live/positions.xlsx"))
    positions_source = "parametro"

    if current_positions is not None:
        cw_qty = dict(current_positions)
    else:
        if positions_path.exists():
            cw_qty = load_positions_from_excel(positions_path)
            positions_source = str(positions_path)
        else:
            cw_qty = {}
            if getattr(cfg, "CREATE_POSITIONS_TEMPLATE_IF_MISSING", True):
                ensure_positions_template(positions_path)
                positions_source = f"plantilla creada en {positions_path}"

    cleaned_qty: dict[str, float] = {}
    for t, q in cw_qty.items():
        p = full_prices_row.get(t, np.nan)
        if pd.notna(p):
            cleaned_qty[t] = float(q)
        else:
            print(f"[posiciones] Aviso: ticker {t!r} sin precio en el ultimo dia — excluido del NAV.")
    cw_qty = cleaned_qty

    nav = portfolio_value_eur(cw_qty, full_prices_row)
    initial = float(getattr(cfg, "INITIAL_CAPITAL", 100_000))
    if nav <= 0 and not cw_qty:
        nav = initial
    elif nav <= 0 and cw_qty:
        print("[posiciones] Aviso: NAV<=0 con posiciones; se usa INITIAL_CAPITAL para el registrador.")
        nav = initial

    cw_weights = weights_from_positions(cw_qty, full_prices_row) if cw_qty else {}

    result = run_pipeline(
        returns_df, prices_df,
        current_weights=cw_weights,
        review_date=last_dt,
    )

    date_str = pd.Timestamp(last_dt).strftime("%Y-%m-%d")
    out_template = getattr(cfg, "REGISTRADOR_OUTPUT_TEMPLATE", "outputs/live/orders.xlsx")
    reg_path = out_template.format(date=date_str)

    reg = run_registrador(
        result["dn_result"],
        market_data,
        current_positions=cw_qty,
        total_value=float(nav),
        output_path=reg_path,
        only_when_engine_would_trade=getattr(
            cfg, "REGISTRADOR_MATCH_ENGINE_REBALANCE_RULE", True
        ),
    )

    post_rebalance_path = None
    ultimo_snapshot_path = None
    if getattr(cfg, "SAVE_SUGGESTED_POSITIONS", True) and reg.get("updated_positions") is not None:
        tpl = (
            getattr(cfg, "POSICIONES_POST_REBALANCEO_TEMPLATE", None)
            or getattr(cfg, "SUGGESTED_POSITIONS_TEMPLATE", "outputs/live/snapshots/positions_{date}.xlsx")
        )
        post_rebalance_path = Path(tpl.format(date=date_str))
        save_positions_to_excel(post_rebalance_path, reg["updated_positions"], as_of_date=last_dt)
        snap_tpl = getattr(cfg, "POSICIONES_ULTIMO_SNAPSHOT_PATH", "outputs/live/snapshots/positions_latest.xlsx")
        ultimo_snapshot_path = Path(snap_tpl)
        save_positions_to_excel(ultimo_snapshot_path, reg["updated_positions"], as_of_date=last_dt)

    if getattr(cfg, "APPEND_EJECUCION_LOG", False):
        log_csv = getattr(cfg, "EJECUCION_LOG_CSV", "outputs/live/execution_log.csv")
        append_ejecucion_row(
            log_csv,
            {
                "fecha_datos": date_str,
                "nav_previo_eur": float(nav),
                "path_operaciones": reg.get("output_path", ""),
                "path_posiciones_post_rebalanceo": str(post_rebalance_path) if post_rebalance_path else "",
                "path_snapshot_ultimo": str(ultimo_snapshot_path) if ultimo_snapshot_path else "",
                "dn_rebalance": result["dn_result"].get("rebalance"),
                "ordenes_filas": len(reg.get("orders", [])),
                "skipped_due_to_dn": reg.get("skipped_due_to_dn", False),
            },
        )

    email_sent = False
    if getattr(cfg, "EMAIL_OPERACIONES_AFTER_RUN", False):
        email_sent = send_operaciones_excel(reg["output_path"], as_of_label=date_str)

    return {
        "market_data": market_data,
        "as_of_date": last_dt,
        "current_positions_qty": cw_qty,
        "current_weights": cw_weights,
        "portfolio_value_eur": float(nav),
        "positions_source": positions_source,
        "posiciones_post_rebalanceo_path": str(post_rebalance_path) if post_rebalance_path else None,
        "posiciones_ultimo_snapshot_path": str(ultimo_snapshot_path) if ultimo_snapshot_path else None,
        "suggested_positions_path": str(post_rebalance_path) if post_rebalance_path else None,
        "email_operaciones_sent": email_sent,
        "registrador": reg,
        **result,
    }


if __name__ == "__main__":
    result = run_single()
    rf = result["risk_free_rate"]
    print(f"Datos hasta: {result['as_of_date']}")
    print(f"Posiciones: {result['positions_source']} — NAV ~ {result['portfolio_value_eur']:,.0f} EUR")
    if result.get("posiciones_post_rebalanceo_path"):
        print(f"Posiciones post-rebalanceo (fecha): {result['posiciones_post_rebalanceo_path']}")
    if result.get("posiciones_ultimo_snapshot_path"):
        print(f"Ultimo snapshot posiciones: {result['posiciones_ultimo_snapshot_path']}")
    if result.get("email_operaciones_sent"):
        print("Correo de operaciones enviado (SMTP configurado).")
    print(f"Tasa BCE: {rf:.2%}")
    print(f"Regimen: {result['regime_early']}")
    print(f"VIEW_SCALE efectivo: {result['effective_view_scale']:.3f}")
    print(f"\nComposite scores (top 10):\n{result['scores'].head(10)}")
    print(f"\nPesos Merton: {result['merton_result']['weights']}")
    print(f"XEON: {result['merton_result']['weight_xeon']:.2%}")
    dn = result["dn_result"]
    print(f"\nDavis-Norman: rebalance={'SI' if dn['rebalance'] else 'NO'} — {dn.get('reason', '')}")
    reg = result["registrador"]
    print(f"\nOperativa guardada en: {reg['output_path']}")
    if reg.get("skipped_due_to_dn"):
        print("(Sin ordenes: Davis-Norman indica no rebalancear — ver hoja Nota en el Excel.)")
    elif len(reg["orders"]) > 0:
        print(reg["orders"].to_string(index=False))
    else:
        print("(Sin filas de ordenes.)")
