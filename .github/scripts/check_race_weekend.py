"""Check whether the pipeline should run for prediction generation.

Two trigger conditions:
  1. Post-race: race just finished for round N -> trigger pipeline for round N+1
     (premature prediction using historical features only, no practice data)
  2. Pre-race: trigger session finished for round N -> trigger pipeline for round N
     Trigger session by weekend type:
       conventional  ->  Practice 3       (Saturday morning)
       sprint        ->  Sprint Qualifying (Friday evening)

Writes should_run, season, round to GITHUB_OUTPUT when the pipeline should run.
"""

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import fastf1


Path("/tmp/fastf1_check_cache").mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache("/tmp/fastf1_check_cache")


def _session_finished(event, session_name, now):
    """check if a session has finished (scheduled start + 2h buffer) and has lap data."""
    try:
        session_start = event.get_session_date(session_name, utc=True)
        if now < session_start + timedelta(hours=2):
            return False
    except Exception:
        return False

    try:
        session = fastf1.get_session(int(event["RoundNumber"]), session_name)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        return session.laps is not None and len(session.laps) > 0
    except Exception:
        return False


def _set_output(year, round_num, label):
    output = f"should_run=true\nseason={year}\nround={round_num}\n"
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(output)
    else:
        print(output)
    print(f"{label}: {year} round {round_num}")


def main():
    today = date.today()
    now = datetime.now(timezone.utc)
    year = today.year

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    total_rounds = len(schedule)

    # check if a race just finished -> premature prediction for next round
    for _, event in schedule.iterrows():
        race_date = event["EventDate"].date()
        if not (race_date <= today <= race_date + timedelta(days=1)):
            continue

        round_num = int(event["RoundNumber"])
        next_round = round_num + 1
        if next_round > total_rounds:
            continue

        # check race session has finished
        try:
            race_start = event.get_session_date("Race", utc=True)
            if now < race_start + timedelta(hours=3):
                continue
        except Exception:
            continue

        # skip if premature prediction already exists for next round
        pred_path = Path(f"reports/predictions/predictions_{year}_{next_round:02d}.json")
        if pred_path.exists():
            print(f"Premature prediction already exists for {year} round {next_round}, skipping.")
            continue

        # confirm race lap data is available
        try:
            session = fastf1.get_session(year, round_num, "Race")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            if session.laps is None or len(session.laps) == 0:
                continue
        except Exception:
            continue

        _set_output(year, next_round, "Post-race premature prediction")
        return

    # check if trigger session finished -> full prediction for current round
    for _, event in schedule.iterrows():
        race_date = event["EventDate"].date()
        if not (race_date - timedelta(days=3) <= today <= race_date):
            continue

        round_num = int(event["RoundNumber"])
        is_sprint = "sprint" in str(event.get("EventFormat", "")).lower()
        trigger_session = "Sprint Qualifying" if is_sprint else "Practice 3"

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

        _set_output(year, round_num, f"Post-{trigger_session} prediction")
        return

    print("No active race weekend or trigger session data not available.")


if __name__ == "__main__":
    main()
