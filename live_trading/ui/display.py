"""Simple display utilities using Rich for terminal UI."""

from typing import List
from rich.table import Table
from rich.panel import Panel
from live_trading.scanner import CityOpportunity
from live_trading.incremental_executor import OrderChunk


def format_opportunities_table(opportunities: List[CityOpportunity]) -> Table:
    """Format opportunities as a Rich table."""
    table = Table(title="Trading Opportunities (Best First)")

    table.add_column("#", style="cyan", justify="right")
    table.add_column("City", style="green")
    table.add_column("Date", style="blue")
    table.add_column("Edge", style="magenta", justify="right")
    table.add_column("Prob", style="yellow", justify="right")
    table.add_column("EV", style="green bold", justify="right")
    table.add_column("Mode", style="white")
    table.add_column("Fcst", style="blue", justify="right")
    table.add_column("Mkt", style="red", justify="right")
    table.add_column("Spread", style="white", justify="right")

    for i, opp in enumerate(opportunities[:15], 1):
        mode_icon = "ML✓" if opp.inference_mode == "classifier" else "THR"

        table.add_row(
            str(i),
            opp.city.upper(),
            str(opp.event_date),
            f"{opp.edge_degf:+.1f}°F",
            f"{opp.edge_classifier_prob:.0%}",
            f"${opp.ev_per_contract/100:.2f}",
            mode_icon,
            f"{opp.forecast_temp:.1f}°F",
            f"{opp.market_implied_temp:.1f}°F",
            f"{opp.spread}¢"
        )

    return table


def format_opportunity_details(opp: CityOpportunity) -> Panel:
    """Format opportunity details as a Rich panel."""
    details = f"""[bold cyan]City:[/bold cyan] {opp.city.upper()}
[bold cyan]Event Date:[/bold cyan] {opp.event_date}
[bold cyan]Ticker:[/bold cyan] {opp.ticker}

[bold yellow]Edge Analysis:[/bold yellow]
  Forecast: {opp.forecast_temp:.1f}°F
  Market:   {opp.market_implied_temp:.1f}°F
  Edge:     {opp.edge_degf:+.1f}°F
  Mode:     {opp.inference_mode.upper()}

[bold yellow]Confidence:[/bold yellow] {opp.edge_classifier_prob:.1%}

[bold green]Market Data:[/bold green]
  Bid: {opp.yes_bid}¢
  Ask: {opp.yes_ask}¢
  Spread: {opp.spread}¢

[bold green]Trade Recommendation:[/bold green]
  Side: {opp.recommended_side.upper()}
  Action: {opp.recommended_action.upper()}
  Price: {opp.recommended_price}¢
  Role: {opp.role.upper()}
  EV: ${opp.ev_per_contract/100:.2f} per contract

[bold]Signal:[/bold] {opp.signal.upper()}
[bold]Reason:[/bold] {opp.reason}
"""
    return Panel(details, title="Opportunity Details", border_style="blue")


def format_execution_plan(chunks: List[OrderChunk], total_ev: float = 0.0) -> Table:
    """Format execution plan as a Rich table."""
    table = Table(title="Execution Plan")

    table.add_column("#", style="cyan", justify="right")
    table.add_column("Contracts", style="yellow", justify="right")
    table.add_column("Price", style="green", justify="right")
    table.add_column("Cost", style="magenta", justify="right")
    table.add_column("Fee", style="red", justify="right")
    table.add_column("Role", style="blue")

    total_cost = 0.0
    total_fee = 0

    for chunk in chunks:
        role = "Maker" if chunk.is_maker else "Taker"

        table.add_row(
            str(chunk.chunk_index),
            str(chunk.num_contracts),
            f"{chunk.price_cents}¢",
            f"${chunk.cost_usd:.2f}",
            f"{chunk.fee_cents}¢",
            role
        )

        total_cost += chunk.cost_usd
        total_fee += chunk.fee_cents

    # Add summary row
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        f"[bold]${total_cost:.2f}[/bold]",
        f"[bold]{total_fee}¢[/bold]",
        ""
    )

    if total_ev > 0:
        table.caption = f"Expected Value: ${total_ev:.2f}"

    return table


def format_positions_table(positions: dict) -> Table:
    """Format open positions as a Rich table."""
    table = Table(title="Open Positions")

    table.add_column("Ticker", style="cyan")
    table.add_column("City", style="green")
    table.add_column("Date", style="blue")
    table.add_column("Side", style="yellow")
    table.add_column("Contracts", style="white", justify="right")
    table.add_column("Entry", style="magenta", justify="right")
    table.add_column("Age", style="white", justify="right")

    for ticker, position in positions.items():
        age_min = int((position.opened_at - position.opened_at).total_seconds() / 60)

        table.add_row(
            ticker,
            position.city.upper(),
            str(position.event_date),
            position.side.upper(),
            str(position.num_contracts),
            f"{position.entry_price_cents}¢",
            f"{age_min}m"
        )

    return table
