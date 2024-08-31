#!/usr/bin/env python
from datetime import datetime
from fluvio import Fluvio

TOPIC_NAME = "python-data"
PARTITION = 0

if __name__ == "__main__":
    # Connect to cluster
    fluvio = Fluvio.connect()

    # Produce 10 records to topic
    producer = fluvio.topic_producer(TOPIC_NAME)
    for x in range(10):
        producer.send_string("{}: timestamp: {}".format(x, datetime.now()))

    # Flush the last entry
    producer.flush()
