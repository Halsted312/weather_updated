"""Simple manual trading CLI with Rich UI."""

import logging

from rich.console import Console
from rich.prompt import Prompt, Confirm

from live_trading.scanner import CityScannerEngine, CityOpportunity
from live_trading.incremental_executor import IncrementalOrderExecutor
from live_trading.position_tracker import PositionTracker
from live_trading.ui.display import (
    format_opportunities_table,
    format_opportunity_details,
    format_execution_plan,
    format_positions_table
)
from src.db.connection import get_db_session

logger = logging.getLogger(__name__)


class ManualTradingCLI:
    """Simple interactive CLI for manual trading."""

    def __init__(
        self,
        scanner: CityScannerEngine,
        executor: IncrementalOrderExecutor,
        position_tracker: PositionTracker,
        dry_run: bool = True
    ):
        self.scanner = scanner
        self.executor = executor
        self.position_tracker = position_tracker
        self.dry_run = dry_run
        self.console = Console()

    async def run(self):
        """Main CLI loop."""
        self.console.print("\n[bold blue]═══ Kalshi Weather Manual Trader ═══[/bold blue]\n")

        if self.dry_run:
            self.console.print("[yellow]⚠ DRY RUN MODE - No real orders will be placed[/yellow]\n")

        while True:
            try:
                # Show menu
                choice = Prompt.ask(
                    "\n[bold]Commands:[/bold] (s)can markets | (p)ositions | (q)uit",
                    choices=['s', 'p', 'q'],
                    default='s'
                )

                if choice == 'q':
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                elif choice == 'p':
                    self._show_positions()
                elif choice == 's':
                    await self._scan_and_trade()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type 'q' to quit.[/yellow]")
                continue
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                logger.error(f"CLI error: {e}", exc_info=True)

    async def _scan_and_trade(self):
        """Scan markets and optionally trade."""
        self.console.print("\n[bold blue]Scanning all 6 cities...[/bold blue]")

        # Scan for opportunities
        with get_db_session() as session:
            opportunities = await self.scanner.scan_all_cities(session)

        if not opportunities:
            self.console.print("[yellow]No trading opportunities found.[/yellow]")
            return

        # Display opportunities
        table = format_opportunities_table(opportunities)
        self.console.print(table)

        # Show summary
        summary = self.scanner.get_opportunity_summary(opportunities)
        self.console.print(f"\n[bold]Found {summary['count']} opportunities, total EV: ${summary['total_ev']:.2f}[/bold]")
        self.console.print(f"[dim]Modes: {summary['inference_modes']}[/dim]")

        # User selection
        choice = Prompt.ask(
            f"\nSelect opportunity (1-{len(opportunities)}) or 'b' to go back",
            default='b'
        )

        if choice == 'b':
            return

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(opportunities):
                self.console.print("[red]Invalid selection[/red]")
                return
        except ValueError:
            self.console.print("[red]Invalid selection[/red]")
            return

        # Show opportunity details
        opp = opportunities[idx]
        panel = format_opportunity_details(opp)
        self.console.print(panel)

        # Check position limits
        can_trade, reason = self.position_tracker.can_open_position(opp.city, opp.event_date)
        if not can_trade:
            self.console.print(f"[red]Cannot trade: {reason}[/red]")
            return

        # Get trade amount
        if not Confirm.ask("\nProceed with trade?"):
            return

        amount_str = Prompt.ask("Enter amount in USD", default="50")
        try:
            amount_usd = float(amount_str)
        except ValueError:
            self.console.print("[red]Invalid amount[/red]")
            return

        # Plan execution
        chunks = self.executor.plan_incremental_entry(
            ticker=opp.ticker,
            target_usd=amount_usd,
            side=opp.recommended_side,
            action=opp.recommended_action,
            yes_bid=opp.yes_bid,
            yes_ask=opp.yes_ask,
        )

        # Show execution plan
        total_ev = opp.ev_per_contract * sum(c.num_contracts for c in chunks) / 100.0
        plan_table = format_execution_plan(chunks, total_ev)
        self.console.print(plan_table)

        # Confirm execution
        if not Confirm.ask("\nExecute this plan?"):
            self.console.print("[yellow]Cancelled[/yellow]")
            return

        # Execute
        await self._execute_trade(opp, chunks)

    async def _execute_trade(self, opp: CityOpportunity, chunks):
        """Execute trade with per-chunk confirmation."""
        self.console.print("\n[bold green]Executing trade...[/bold green]")

        if self.dry_run:
            self.console.print("[yellow]DRY RUN - Would place orders[/yellow]")
            for chunk in chunks:
                self.console.print(f"  Would place: {chunk}")
            return

        # Callback for confirmation
        async def confirm_callback(message: str, chunk) -> bool:
            return Confirm.ask(message)

        # Execute
        placed = await self.executor.execute_incremental_order(
            ticker=opp.ticker,
            city=opp.city,
            event_date=opp.event_date,
            side=opp.recommended_side,
            action=opp.recommended_action,
            yes_bid=opp.yes_bid,
            yes_ask=opp.yes_ask,
            chunks=chunks,
            confirm_each=True,
            callback=confirm_callback,
        )

        # Report results
        self.console.print(f"\n[bold green]✓ Placed {len(placed)} orders[/bold green]")
        for order_id, chunk in placed:
            self.console.print(f"  Order: {order_id}")

        # Add position
        if placed:
            total_contracts = sum(chunk.num_contracts for _, chunk in placed)
            self.position_tracker.add_position(
                ticker=opp.ticker,
                city=opp.city,
                event_date=opp.event_date,
                side=opp.recommended_side,
                num_contracts=total_contracts,
                entry_price_cents=opp.recommended_price,
            )

    def _show_positions(self):
        """Show open positions."""
        if not self.position_tracker.positions:
            self.console.print("\n[yellow]No open positions[/yellow]")
            return

        table = format_positions_table(self.position_tracker.positions)
        self.console.print(f"\n{table}")

        # Show P&L
        total_pnl = self.position_tracker.get_total_pnl_usd()
        daily_pnl = self.position_tracker.get_daily_pnl_usd()

        self.console.print(f"\n[bold]Today's P&L:[/bold] ${daily_pnl:+.2f}")
        self.console.print(f"[bold]Total P&L:[/bold] ${total_pnl:+.2f}")
