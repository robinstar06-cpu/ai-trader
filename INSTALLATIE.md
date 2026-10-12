# Installatie (± 10 minuten)

## 1. Alpaca-demoaccount van $10.000
1. Ga naar het Alpaca-dashboard en klik linksboven op je **paper-accountnummer**.
2. Kies **Open New Paper Account** en stel het startbedrag in op **$10.000**.
3. Maak **voor dit nieuwe account** nieuwe API-sleutels aan (Home → API Keys → Generate New Keys).
   Je oude sleutels horen bij het account van $100.000.

## 2. Repository op GitHub
1. Ga naar <https://github.com/new>.
2. Naam: `ai-trader`. Kies **Public**: dan zijn de GitHub-minuten onbeperkt gratis. Je sleutels blijven geheim (stap 3); alleen de code en het logboek zijn zichtbaar.
3. Klik **Create repository** en daarna op **uploading an existing file**.
4. Pak de zip uit en sleep **alle bestanden en mappen** in het venster, ook de map `.github`. Klik **Commit changes**.

> Zie je bij stap 5 geen "AI Trader" onder Actions? Dan is de map `.github` niet meegekomen.
> Kies dan **Add file → Create new file**, typ als naam `.github/workflows/trader.yml` en plak de inhoud van dat bestand erin.

## 3. Sleutels toevoegen (blijven geheim)
Ga naar **Settings → Secrets and variables → Actions → New repository secret** en maak deze drie aan:

| Name | Waarde |
|---|---|
| `ALPACA_API_KEY` | API Key ID van Alpaca (begint met `PK`) |
| `ALPACA_SECRET_KEY` | Secret Key van Alpaca |
| `NVIDIA_API_KEY` | je NVIDIA-sleutel (begint met `nvapi-`) |

## 4. Testen
1. Ga naar het tabblad **Actions**, kies **AI Trader** en klik op **Run workflow**.
2. Vink **Proefrun** aan en klik **Run workflow**. De bot doet dan alles behalve orders plaatsen.
3. Na 1 à 3 minuten staat er een groen vinkje. Klik op de run om het verslag te zien.
4. Ziet het er goed uit? Dan hoef je niets meer te doen: de bot draait vanaf nu **elk half uur** vanzelf.

## Resultaten bekijken
- De **startpagina van je repository** (README) toont na elke run de waarde, het rendement tegenover BTC vasthouden, de posities en de redenen van de AI.
- Volledig logboek: `data/equity.csv` en `data/decisions.csv`.
- Je kunt ook in het Alpaca-dashboard kijken.

## Problemen
- **Rode run met "Permission denied" bij "Logboek opslaan":** Settings → Actions → General → Workflow permissions → **Read and write permissions** → Save.
- **"Ontbrekende sleutels":** controleer of de namen in stap 3 exact kloppen.
- **Stoppen:** Actions → AI Trader → **⋯** → **Disable workflow**.

## Instellingen (optioneel)
Via **Settings → Secrets and variables → Actions → tab Variables**:

| Variable | Standaard | Betekenis |
|---|---|---|
| `MODELS` | `moonshotai/kimi-k3,z-ai/glm-5.3,deepseek-ai/deepseek-v4.1-flash` | AI-modellen in volgorde (reserve bij uitval) |
| `SYMBOLS` | `ALL` | Welke munten, bijvoorbeeld `BTC/USD,ETH/USD` |
| `GOAL_PCT` | `20` | Doel per periode (% winst) |
| `GOAL_DAYS` | `28` | Lengte van een periode in dagen |
| `GOAL_STEP_PCT` | `10` | Doel gehaald? Dan gaat de lat zoveel % omhoog |
| `RISK_TOLERANCE_PCT` | `25` | Daling vanaf de top die de AI mag accepteren |
| `MAX_PER_COIN_PCT` | `60` | Maximaal % per munt |
| `MIN_CASH_PCT` | `0` | Minimaal % cash |
| `REBALANCE_BAND_PCT` | `1` | Wijzigingen kleiner dan dit % negeren |
| `STOP_BUYING_DRAWDOWN_PCT` | `40` | Noodrem: bij zoveel % verlies stopt de bot met kopen |

Vaker of minder vaak draaien: pas de regel `cron` aan in `.github/workflows/trader.yml`.
