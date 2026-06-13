"""Check whether the pipeline should run for prediction generation.

Three trigger conditions (checked in priority order):
  1. Post-race: race just finished for round N -> trigger pipeline for round N+1
     (preliminary prediction using historical features only, no practice data)
  2. Pre-race: trigger session finished for round N -> trigger pipeline for round N
     Trigger session by weekend type:
       conventional  ->  Practice 3       (Saturday morning)
       sprint        ->  Sprint Qualifying (Friday evening)
  3. Post-FP2: Practice 2 just finished for round N (conventional weekends only)
     -> trigger pipeline for round N (preliminary prediction with early practice data)

Writes should_run, season, round to GITHUB_OUTPUT when the pipeline should run.
"""

import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import fastf1


Path("/tmp/fastf1_check_cache").mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache("/tmp/fastf1_check_cache")


def _set_output(year, round_num, trigger, label):
    output = f"should_run=true\nseason={year}\nround={round_num}\ntrigger={trigger}\n"
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(output)
    else:
        print(output)
    print(f"{label}: {year} round {round_num}")


def _commit_exists(pattern):
    """Check if a commit matching the pattern exists in recent history."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--grep", pattern, "-1"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


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
        print(f"Post-race check: round {round_num}, next {next_round}, race_date {race_date}")

        if next_round > total_rounds:
            print(f"Round {next_round} exceeds total rounds ({total_rounds}), skipping.")
            continue

        # check race session has finished
        try:
            race_start = event.get_session_date("Race", utc=True)
            if race_start.tzinfo is None:
                race_start = race_start.replace(tzinfo=timezone.utc)
            if now < race_start + timedelta(hours=3):
                print(f"Race not finished yet (start {race_start}, need +3h).")
                continue
        except Exception as e:
            print(f"Could not get Race session date: {e}", file=sys.stderr)
            continue

        # skip if premature prediction already exists for next round
        pred_path = Path(f"reports/predictions/predictions_{year}_{next_round:02d}.json")
        if pred_path.exists():
            print(f"Premature prediction already exists for {year} round {next_round}, skipping.")
            continue

        _set_output(year, next_round, "post-race", "Post-race premature prediction")
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
            if session_start.tzinfo is None:
                session_start = session_start.replace(tzinfo=timezone.utc)
            if now < session_start + timedelta(hours=2):
                # trigger session not ready yet, fall through to FP2 check
                break
        except Exception as e:
            print(f"Could not get {trigger_session} time: {e}", file=sys.stderr)
            break

        dedup_msg = f"Post-FP3 predictions for {year} round {round_num}"
        if _commit_exists(dedup_msg):
            print(f"Already committed: {dedup_msg}, skipping.")
            break

        _set_output(year, round_num, "pre-race", f"Post-{trigger_session} prediction")
        return

    # check if FP2 finished -> preliminary prediction with early practice data
    for _, event in schedule.iterrows():
        race_date = event["EventDate"].date()
        if not (race_date - timedelta(days=3) <= today <= race_date):
            continue

        round_num = int(event["RoundNumber"])
        is_sprint = "sprint" in str(event.get("EventFormat", "")).lower()
        if is_sprint:
            print("Sprint weekend, no FP2 trigger.")
            continue

        try:
            fp2_start = event.get_session_date("Practice 2", utc=True)
            if fp2_start.tzinfo is None:
                fp2_start = fp2_start.replace(tzinfo=timezone.utc)
            if now < fp2_start + timedelta(hours=2):
                print("Practice 2 not finished yet.")
                continue
        except Exception as e:
            print(f"Could not get Practice 2 time: {e}", file=sys.stderr)
            continue

        commit_msg = f"Post-FP2 preliminary predictions for {year} round {round_num}"
        if _commit_exists(commit_msg):
            print(f"Already committed: {commit_msg}, skipping.")
            continue

        _set_output(year, round_num, "post-fp2", "Post-FP2 preliminary prediction")
        return

    print("No active race weekend or trigger session data not available.")


if __name__ == "__main__":
    main()
