"""
Tests for weather data clients.

Includes:
- Visual Crossing client tests
- NWS CLI client tests
- NWS CF6 client tests
- Integration tests (require API keys or network access)
"""

import re
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.config import CITIES, EXCLUDED_VC_CITIES, get_settings
from src.weather.nws_cli import SETTLEMENT_STATIONS, NWSCliClient, get_settlement_station
from src.weather.nws_cf6 import NWSCF6Client


class TestSettlementStations:
    """Test settlement station configuration."""

    def test_all_cities_have_stations(self):
        """Verify all non-excluded cities have settlement station config."""
        for city_id in CITIES.keys():
            if city_id not in EXCLUDED_VC_CITIES:
                assert city_id in SETTLEMENT_STATIONS, f"{city_id} missing from SETTLEMENT_STATIONS"

    def test_station_has_required_fields(self):
        """Verify each station has all required fields."""
        for city_id, station in SETTLEMENT_STATIONS.items():
            assert "icao" in station, f"{city_id} missing icao"
            assert "issuedby" in station, f"{city_id} missing issuedby"
            assert "ghcnd" in station, f"{city_id} missing ghcnd"

    def test_icao_format(self):
        """Verify ICAO codes are 4 characters starting with K."""
        for city_id, station in SETTLEMENT_STATIONS.items():
            icao = station["icao"]
            assert len(icao) == 4, f"{city_id} ICAO {icao} not 4 chars"
            assert icao.startswith("K"), f"{city_id} ICAO {icao} doesn't start with K"

    def test_ghcnd_format(self):
        """Verify GHCND station IDs have correct format."""
        for city_id, station in SETTLEMENT_STATIONS.items():
            ghcnd = station["ghcnd"]
            assert ghcnd.startswith("GHCND:"), f"{city_id} GHCND {ghcnd} doesn't start with GHCND:"
            assert len(ghcnd) > 10, f"{city_id} GHCND {ghcnd} too short"

    def test_get_settlement_station(self):
        """Test get_settlement_station helper."""
        station = get_settlement_station("chicago")
        assert station is not None
        assert station["icao"] == "KMDW"
        assert station["issuedby"] == "MDW"

        # Unknown city should return None
        assert get_settlement_station("unknown_city") is None


class TestNWSCliClient:
    """Test NWS CLI client."""

    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = NWSCliClient()
        assert client.session is not None
        assert "User-Agent" in client.session.headers

    def test_parse_cli_tmax_valid(self):
        """Test parsing valid CLI HTML."""
        client = NWSCliClient()

        # Sample CLI HTML (simplified)
        sample_html = """
        <pre class="glossaryProduct">
        CLIMATE SUMMARY FOR NOVEMBER 25 2025

        TEMPERATURE (DEGREES FAHRENHEIT)
                           TODAY   NORMAL  RECORD
        MAXIMUM         42   4:55 PM   48     72 (2024)

        PRECIPITATION (INCHES)
        </pre>
        """

        result = client.parse_cli_tmax(sample_html)

        assert result is not None
        assert result["date_local"] == date(2025, 11, 25)
        assert result["tmax_f"] == 42.0
        assert result["is_preliminary"] is False
        assert result["source"] == "CLI"

    def test_parse_cli_tmax_with_target_date(self):
        """Test parsing CLI with target date validation."""
        client = NWSCliClient()

        sample_html = """
        <pre class="glossaryProduct">
        CLIMATE SUMMARY FOR NOVEMBER 25 2025

        MAXIMUM         42   4:55 PM   48     72 (2024)
        </pre>
        """

        # Matching date should work
        result = client.parse_cli_tmax(sample_html, target_date=date(2025, 11, 25))
        assert result is not None

        # Non-matching date should return None
        result = client.parse_cli_tmax(sample_html, target_date=date(2025, 11, 24))
        assert result is None

    def test_parse_cli_tmax_invalid_html(self):
        """Test parsing invalid HTML returns None."""
        client = NWSCliClient()

        # No pre tag
        result = client.parse_cli_tmax("<html><body>No data</body></html>")
        assert result is None

        # No CLIMATE SUMMARY
        result = client.parse_cli_tmax("<pre class='glossaryProduct'>Random text</pre>")
        assert result is None

    def test_get_tmax_unknown_city(self):
        """Test get_tmax_for_city raises error for unknown city."""
        client = NWSCliClient()

        with pytest.raises(ValueError, match="Unknown city"):
            client.get_tmax_for_city("unknown_city")


class TestNWSCF6Client:
    """Test NWS CF6 client."""

    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = NWSCF6Client()
        assert client.session is not None
        assert "User-Agent" in client.session.headers

    def test_parse_cf6_tmax_valid(self):
        """Test parsing valid CF6 HTML."""
        client = NWSCF6Client()

        # Sample CF6 HTML (simplified)
        sample_html = """
        <pre class="glossaryProduct">
        PRELIMINARY LOCAL CLIMATOLOGICAL DATA (WS FORM: F-6)

                                       STATION: CHICAGO MIDWAY AIRPORT IL
                                       MONTH:   NOVEMBER
                                       YEAR:    2025

        DY MAX MIN AVG DEP HDD CDD

         1  52  41  47   0   18   0
         2  55  43  49   2   16   0
        11  42  25  34  -7   31   0
        25  38  28  33  -5   32   0

        ===============================================
        </pre>
        """

        result = client.parse_cf6_tmax(sample_html, target_date=date(2025, 11, 25))

        assert result is not None
        assert result["date_local"] == date(2025, 11, 25)
        assert result["tmax_f"] == 38.0
        assert result["is_preliminary"] is True
        assert result["source"] == "CF6"

    def test_parse_cf6_tmax_wrong_month(self):
        """Test CF6 parsing rejects wrong month."""
        client = NWSCF6Client()

        sample_html = """
        <pre class="glossaryProduct">
        MONTH:   NOVEMBER
        YEAR:    2025

        DY MAX MIN AVG
        25  38  28  33
        </pre>
        """

        # Should reject because target is October, not November
        result = client.parse_cf6_tmax(sample_html, target_date=date(2025, 10, 25))
        assert result is None

    def test_parse_cf6_tmax_missing_day(self):
        """Test CF6 parsing handles missing day."""
        client = NWSCF6Client()

        sample_html = """
        <pre class="glossaryProduct">
        MONTH:   NOVEMBER
        YEAR:    2025

        DY MAX MIN AVG
         1  52  41  47
         2  55  43  49
        </pre>
        """

        # Day 25 not in table
        result = client.parse_cf6_tmax(sample_html, target_date=date(2025, 11, 25))
        assert result is None

    def test_get_tmax_unknown_city(self):
        """Test get_tmax_for_city raises error for unknown city."""
        client = NWSCF6Client()

        with pytest.raises(ValueError, match="Unknown city"):
            client.get_tmax_for_city("unknown_city", date.today())


class TestVisualCrossingClient:
    """Test Visual Crossing client."""

    def test_client_initialization(self):
        """Test client initializes correctly."""
        from src.weather.visual_crossing import VisualCrossingClient

        client = VisualCrossingClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.minute_interval == 5
        assert client.session is not None

    def test_format_list(self):
        """Test _format_list helper."""
        from src.weather.visual_crossing import VisualCrossingClient

        client = VisualCrossingClient(api_key="test_key")

        assert client._format_list(["rain", "snow"]) == "rain,snow"
        assert client._format_list([]) is None
        assert client._format_list("single") == "single"
        assert client._format_list(None) is None

    def test_excluded_cities_exist(self):
        """Test that EXCLUDED_VC_CITIES is properly configured."""
        # NYC should be in the excluded list (due to high forward-fill)
        assert "new_york" in EXCLUDED_VC_CITIES

    def test_valid_cities_not_excluded(self):
        """Test that our 6 target cities are not excluded."""
        from src.weather.visual_crossing import VisualCrossingClient

        valid_cities = ["chicago", "austin", "denver", "los_angeles", "miami", "philadelphia"]
        for city in valid_cities:
            assert city not in EXCLUDED_VC_CITIES, f"{city} should not be excluded"


@pytest.mark.integration
class TestNWSCliIntegration:
    """
    Integration tests for NWS CLI client.

    Run with: pytest -m integration tests/test_weather_clients.py
    """

    def test_fetch_chicago_cli(self):
        """Test fetching CLI for Chicago."""
        client = NWSCliClient()

        result = client.get_tmax_for_city("chicago")

        # May be None if CLI not yet posted today
        if result:
            assert result["city_id"] == "chicago"
            assert result["icao"] == "KMDW"
            assert result["issuedby"] == "MDW"
            assert isinstance(result["tmax_f"], float)
            assert isinstance(result["date_local"], date)
            assert result["is_preliminary"] is False

    def test_fetch_all_cities_cli(self):
        """Test fetching CLI for all cities."""
        client = NWSCliClient()

        results = client.fetch_all_cities()

        # Should have results for all configured cities
        assert len(results) == len(SETTLEMENT_STATIONS)

        for city_id, result in results.items():
            if result:
                assert result["city_id"] == city_id
                assert "tmax_f" in result


@pytest.mark.integration
class TestNWSCF6Integration:
    """
    Integration tests for NWS CF6 client.

    Run with: pytest -m integration tests/test_weather_clients.py
    """

    def test_fetch_chicago_cf6(self):
        """Test fetching CF6 for Chicago for yesterday."""
        client = NWSCF6Client()

        yesterday = date.today() - timedelta(days=1)
        result = client.get_tmax_for_city("chicago", yesterday)

        # May be None if day not yet in CF6
        if result:
            assert result["city_id"] == "chicago"
            assert result["icao"] == "KMDW"
            assert isinstance(result["tmax_f"], float)
            assert result["is_preliminary"] is True


@pytest.mark.integration
class TestVisualCrossingIntegration:
    """
    Integration tests for Visual Crossing client.

    Run with: pytest -m integration tests/test_weather_clients.py
    """

    @pytest.fixture
    def client(self):
        """Create client with real credentials."""
        from src.weather.visual_crossing import VisualCrossingClient

        settings = get_settings()

        if not settings.vc_api_key:
            pytest.skip("Visual Crossing API key not configured")

        return VisualCrossingClient(
            api_key=settings.vc_api_key,
            base_url=settings.vc_base_url,
            minute_interval=settings.wx_minute_interval,
        )

    def test_fetch_chicago_minutes(self, client):
        """Test fetching minute data for Chicago."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        df = client.fetch_day_for_city("chicago", yesterday)

        if not df.empty:
            # Should have many minute records for a day
            assert len(df) > 100  # ~288 records for 5-min intervals

            # Check required columns exist
            required_cols = [
                "ts_utc", "temp_f", "humidity", "dew_f",
                "windspeed_mph", "pressure_mb", "stations",
            ]
            for col in required_cols:
                assert col in df.columns, f"Missing column: {col}"

            # Check temp values are reasonable
            assert df["temp_f"].min() > -50  # Not colder than -50F
            assert df["temp_f"].max() < 150  # Not hotter than 150F

    def test_fetch_all_elements_present(self, client):
        """Test that all weather elements are captured."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        df = client.fetch_day_for_city("chicago", yesterday)

        if not df.empty:
            # Check ALL elements are captured (future-proofing)
            expected_cols = [
                "ts_utc",
                # Temperature
                "temp_f", "feelslike_f", "dew_f",
                # Humidity
                "humidity",
                # Precipitation
                "precip_in", "precip_prob", "precip_type",
                "snow_in", "snow_depth_in",
                # Wind
                "windspeed_mph", "winddir_deg", "windgust_mph",
                # Pressure/visibility
                "pressure_mb", "visibility_mi",
                # Cloud/solar
                "cloud_cover", "solar_radiation", "solar_energy", "uv_index",
                # Conditions
                "conditions", "icon",
                # Audit
                "stations", "raw_json",
            ]

            for col in expected_cols:
                assert col in df.columns, f"Missing expected column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
