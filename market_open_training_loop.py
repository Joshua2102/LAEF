import datetime
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Set timezone offset if needed (e.g., EST is UTC-5, but you may need to use local time directly)
MARKET_OPEN = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)


def is_market_open():
    now = datetime.datetime.now()
    return now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE


def wait_for_market_open():
    logging.info("Waiting for market to open...")
    while True:
        now = datetime.datetime.now()
        if now.weekday() >= 5:  # Weekend
            logging.info("It's the weekend. Sleeping 12 hours.")
            time.sleep(43200)  # 12 hours
            continue

        if now.time() < MARKET_OPEN:
            logging.info("Market not open yet. Sleeping 5 minutes.")
            time.sleep(300)
        elif now.time() > MARKET_CLOSE:
            logging.info("Market closed. Sleeping until tomorrow.")
            time.sleep(3600)
        else:
            logging.info("Market is open.")
            return


def train_during_market():
    logging.info("Starting training loop during market hours...")
    while is_market_open():
        logging.info("Running trainer.py...")
        subprocess.call(["python", "trainer.py"])
        logging.info("Trainer run complete. Sleeping 15 minutes.")
        time.sleep(900)  # 15 min
    logging.info("Market closed. Ending training loop.")


def main():
    while True:
        wait_for_market_open()
        train_during_market()
        logging.info("Sleeping 8 hours until next check cycle...")
        time.sleep(28800)  # Sleep 8 hrs to reduce CPU cycles after market close


if __name__ == "__main__":
    main()
