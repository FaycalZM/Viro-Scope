from fluvio import Fluvio, Offset


TOPIC_NAME = "google-trends-data"
PARTITION = 0

if __name__ == "__main__":
    # Connect to cluster
    fluvio = Fluvio.connect()

    # Consume last 10 records from topic
    consumer = fluvio.partition_consumer(TOPIC_NAME, PARTITION)
    output = ""
    i = 0
    stream = consumer.stream(Offset.from_beginning(0))
    for record in stream:
        if record.value_string() == "done":
            break
        else:
            output += record.value_string()
            i += 1

    print(f"Finished consuming all {i} available messages.")
    print(output)
