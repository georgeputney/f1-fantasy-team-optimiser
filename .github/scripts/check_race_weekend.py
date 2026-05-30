"""Check whether the trigger session for the current race weekend has completed.

Trigger session by weekend type:
  conventional  ->  Practice 3       (Saturday morning)
  sprint        ->  Sprint Qualifying (Friday evening)

Writes should_run, season, round to GITHUB_OUTPUT when the pipeline should run.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import fastf1


Path("/tmp/fastf1_check_cache").mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache("/tmp/fastf1_check_cache")


def _already_predicted(year: int, round_num: int) -> bool:
    pred_path = Path("reports/predictions_latest.json")
    if not pred_path.exists():
        return False
    try:
        data = json.loads(pred_path.read_text())
        return data.get("season") == year and data.get("round") == round_num
    except Exception:
        return False


def main():
    today = date.today()
    now = datetime.now(timezone.utc)
    year = today.year

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for _, event in schedule.iterrows():
        race_date = event["EventDate"].date()
        if not (race_date - timedelta(days=3) <= today <= race_date):
            continue

        round_num = int(event["RoundNumber"])
        is_sprint = "sprint" in str(event.get("EventFormat", "")).lower()
        trigger_session = "Sprint Qualifying" if is_sprint else "Practice 3"

        if _already_predicted(year, round_num):
            print(f"Predictions already exist for {year} round {round_num}, skipping.")
            return

        # check trigger session has finished (scheduled start + 2h buffer)
        try:
            session_start = event.get_session_date(trigger_session, utc=True)
            if now < session_start + timedelta(hours=2):
                print(f"{trigger_session} not finished yet.")
                return
        except Exception as e:
            print(f"Could not get {trigger_session} time: {e}", file=sys.stderr)
            return

        # confirm lap data is in FastF1
        try:
            session = fastf1.get_session(year, round_num, trigger_session)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            if session.laps is None or len(session.laps) == 0:
                print(f"{trigger_session} laps not yet available.")
                return
        except Exception as e:
            print(f"{trigger_session} data not available: {e}", file=sys.stderr)
            return

        output = f"should_run=true\nseason={year}\nround={round_num}\n"
        output_file = os.environ.get("GITHUB_OUTPUT", "")
        if output_file:
            with open(output_file, "a") as f:
                f.write(output)
        else:
            print(output)

        print(f"Ready: {year} round {round_num} — {event['EventName']}")
        return

    print("No active race weekend or trigger session data not available.")


if __name__ == "__main__":
    main()
